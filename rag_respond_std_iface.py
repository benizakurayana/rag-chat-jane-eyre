from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_chroma import Chroma
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage
from typing import Dict
from langchain_core.runnables import RunnablePassthrough
from utils.env_setup import setup_environment
from preprocess import preprocess


# Set environment variables
setup_environment()

# Download the novel, split into chunks and create vectorstore
preprocess()


def initialize_components():
    """
    Initialize RAG components: chat model, retriever
    :return: VectorStoreRetriever, ChatOpenAI
    """
    # Set the embedding model
    embeddings = OpenAIEmbeddings(model="text-embedding-3-large")  # TODO: one of the environment variables

    # Load the persisted vector store
    persist_directory = 'chroma_db/paragraphs'
    loaded_vectorstore = Chroma(
        persist_directory=persist_directory,
        collection_name='paragraphs',
        embedding_function=embeddings,
        create_collection_if_not_exists=False
    )

    # Create retriever
    retriever = loaded_vectorstore.as_retriever(
        search_type="mmr",
        search_kwargs={
            "k": 5,  # Retrieve top 5 chunks
            "fetch_k": 20,  # Fetch 20 chunks for re-ranking
            "maximal_marginal_relevance": True,  # Diversify results
            "lambda_mult": 0.7  # Prioritize relevance slightly more than diversity
        }
    )

    # Create chat model
    chat = ChatOpenAI(model="gpt-4o-mini", temperature=0.2)  # TODO: one of the environment variables

    return retriever, chat


# System template
SYSTEM_TEMPLATE = """
Answer the user's questions based on the below context. 
If the context doesn't contain any relevant information to the question, don't make something up and just say "I don't know."
Answer in the user's input language.

<context>
{context}
</context>
"""


def create_chain(retriever, chat):
    """
    Create the retrieval chain with given components
    :param retriever: VectorStoreRetriever
    :param chat: ChatOpenAI
    :return: RunnablePassthrough, the retrieval chain
    """
    # Create document chain (a component of the retrieval chain)
    question_answering_prompt = ChatPromptTemplate.from_messages([
        ("system", SYSTEM_TEMPLATE),
        MessagesPlaceholder(variable_name="messages"),
    ])
    document_chain = create_stuff_documents_chain(chat, question_answering_prompt)
    """
    The "create_stuff_documents_chain" function:
    - Takes all retrieved documents
    - Concatenates their content into one context
    - "Stuffs" all this context into a single prompt
    Process:
    - Input Documents → Concatenate (into a string) → Insert into Template → Send to Model → Get Answer
    - [Doc1, Doc2, Doc3] → "Doc1 + Doc2 + Doc3" → Template with Context → GPT-4 → Response
    """

    # Create retrieval chain (parent of document chain. higher-level)
    def parse_retriever_input(params: Dict):
        """
        This helper function extracts the actual question text from the input message structure
        It's used as a preprocessing step in the retrieval chain before passing the query to the retriever
        :param params: Dict
        :return: str, the question string
        """
        return params["messages"][-1].content

    retrieval_chain = RunnablePassthrough.assign(  # 1st Assignment
        context=parse_retriever_input | retriever,
    ).assign(                                      # 2nd Assignment
        answer=document_chain,
    )
    """
    RunnablePassthrough class:
    - A LangChain utility class that helps create sequential processing chains
    - Allows you to "pass through" data while adding or transforming fields
    - Each .assign() adds a new field to the output dictionary

    1st Assignment (context): context=parse_retriever_input | retriever
    - The | operator creates a pipeline:
        1. parse_retriever_input runs first:
            Input: {"messages": [HumanMessage(content="what is wuthering heights")]}
            Output: "what is wuthering heights"
        2. retriever runs second:
            Takes the extracted question
            Returns relevant documents
    - Result stored in 'context' key of output dictionary

    2nd Assignment (answer): answer=document_chain
    - Takes the current state (including context from first assignment)
    - Passes it through the document_chain
    - Stores result in 'answer' key of output dictionary
    """

    """
    Process:
    HIGH-LEVEL FLOW:
    INPUT MESSAGE 
    → FIRST ASSIGN (parse_retriever_input | retriever) 
    → SECOND ASSIGN (document_chain)
    → FINAL OUTPUT

    DETAILED FLOW:
    1. INITIAL STATE
        INPUT MESSAGE : {"messages": [HumanMessage(content="what is wuthering heights")]}

    2. FIRST ASSIGN
       a. parse_retriever_input (Parse Question)
          INPUT:  {"messages": [HumanMessage("what is wuthering heights")]}
          OUTPUT: "what is wuthering heights"

       b. retriever (Fetch Documents)
          INPUT:  "what is wuthering heights"
          OUTPUT: [Document1, Document2, ...]

       INTERMEDIATE STATE: {
          "messages": [HumanMessage(content="what is wuthering heights")],
          "context": [Document1, Document2, ...]
       }

    3. SECOND ASSIGN
       document_chain (Generate Answer)
       INPUT: {
          "context": [Document1, Document2],
          "messages": [HumanMessage("what is wuthering heights")]
       }
       OUTPUT: "The Wuthering Heights is..."

    4. FINAL STATE
        FINAL OUTPUT: {
            "messages": [...],
            "context": [Document1, Document2, ...],
            "answer": "The Wuthering Heights is..."
        }
    """

    return retrieval_chain


# Initialize components and chain at module level
_retriever, _chat = initialize_components()
_chain = create_chain(_retriever, _chat)


def rag_respond_std_iface(question):
    """
    Process a question and return an answer using the retrieval chain
    :param question: str, the question provided by the user.
    :return: str, the generated answer extracted from the retrieval chain's response.
    """
    response = _chain.invoke({
        "messages": [HumanMessage(content=question)],
    })

    return response['answer']

