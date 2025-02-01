import os
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_chroma import Chroma
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.runnables import RunnablePassthrough
from utils.env_setup import setup_environment
from preprocess import preprocess
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableBranch


# Set environment variables
setup_environment()
LLM_DEPLOYMENT_NAME = os.getenv('LLM_DEPLOYMENT_NAME')
EMBEDDING_DEPLOYMENT_NAME = os.getenv('EMBEDDING_DEPLOYMENT_NAME')

# Download the novel, split into chunks and create vectorstore
preprocess()


def initialize_components():
    """
    Initialize RAG components: chat model, retriever
    :return: VectorStoreRetriever, ChatOpenAI
    """
    # Set the embedding model
    embeddings = OpenAIEmbeddings(model=EMBEDDING_DEPLOYMENT_NAME)

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
    chat = ChatOpenAI(model=LLM_DEPLOYMENT_NAME, temperature=0.2)

    return retriever, chat


# System template
SYSTEM_TEMPLATE = """
You are an assistant for question-answering tasks. 
Use the following pieces of retrieved context to answer the user's question. 
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
    :return: RunnablePassthrough, the conversational retrieval chain
    """
    """
    Process:
    HIGH-LEVEL FLOW:
    INPUT MESSAGE 
    → FIRST ASSIGN (query_transforming_retriever_chain) 
    → SECOND ASSIGN (document_chain)
    → FINAL OUTPUT

    DETAILED FLOW [Case 1]:
    Almost the same with the flow of rag_respond_std_iface.py
    1. INITIAL STATE
        INPUT MESSAGE : {"messages": [HumanMessage(content="who is jane eyre?")]}

    2. FIRST ASSIGN
       a. lambda x: x["messages"][-1].content (Parse Question)
          INPUT:  {"messages": [HumanMessage("who is jane eyre?")]}
          OUTPUT: "who is jane eyre?"

       b. retriever (Fetch Documents)
          INPUT:  "who is jane eyre?"
          OUTPUT: [Document1, Document2, ...]

       INTERMEDIATE STATE : {
          "messages": [HumanMessage(content="who is jane eyre?")],
          "context": [Document1, Document2, ...]
       }

    3. SECOND ASSIGN
       document_chain (Generate Answer)
       INPUT: {
          "messages": [HumanMessage("who is jane eyre?")],
          "context": [Document1, Document2, ...]
       }
       OUTPUT: "Jane Eyre is..."     

    4. FINAL STATE
       FINAL OUTPUT: {
            "messages": [...],
            "context": [Document1, Document2, ...],
            "answer": "Jane Eyre is..."
       }

    -----------------------------------------------------------------    
    DETAILED FLOW [Case 2]:
    1. INITIAL STATE
        INPUT MESSAGE : {"messages": [HumanMessage(content="who is jane eyre?"), 
                                                AIMessage(content="Jane Eyre is..."),
                                                HumanMessage(content="what happened in her childhood?")]}

    2. FIRST ASSIGN
       a. query_transform_prompt (Insert the chat history into the prompt)
       b. chat (Generate query string for retriever)
       c. StrOutputParser() (Parse the output from the chat model)
       d. retriever (Fetch Documents)

       INTERMEDIATE STATE: {
          "messages": [HumanMessage(content="who is jane eyre?"), 
                        AIMessage(content="Jane Eyre is..."),
                        HumanMessage(content="what happened in her childhood?")],
          "context": [Document3, Document4, ...]
       }

    3. SECOND ASSIGN
       document_chain (Generate Answer)
       INPUT: {
          "messages": [HumanMessage(content="who is jane eyre?"), 
                        AIMessage(content="Jane Eyre is..."),
                        HumanMessage(content="what happened in her childhood?")],
          "context": [Document3, Document4, ...]
       }
       OUTPUT: "When Jane Eyre was young..."

    4. FINAL STATE
        FINAL OUTPUT: {
            "messages": [...],
            "context": [Document3, Document4, ...],
            "answer": "When Jane Eyre was young..."
        }
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
    - Input documents → Concatenate (into a string) → Insert into template → Send to chat model → Get answer
    - [Doc1, Doc2, Doc3] → "Doc1 Doc2 Doc3" → Template with context → GPT-4 → Response
    """


    # Create query transformation prompt, used in transforming retriever chain
    query_transform_prompt = ChatPromptTemplate.from_messages(
        [
            MessagesPlaceholder(variable_name="messages"),
            (
                "user",
                "Given the above conversation, generate a query to look up in order to get information relevant "
                "to the conversation. The search will be used in a vectorstore of the full text of a novel. "
                "In the novel, the narrator (I) is 'Jane Eyre'. "
                "Only respond with the query, nothing else..",
            ),
        ]
    )

    # Create query transforming retriever chain (a component of the conversational retrieval chain)
    query_transforming_retriever_chain = RunnableBranch(
        (
            # If only one message, then we just pass that message's content to retriever
            lambda x: len(x.get("messages", [])) == 1,
            (lambda x: x["messages"][-1].content) | retriever,
        ),
        # Else (if messages), then we pass inputs to LLM chain to transform the query, then pass to retriever
        query_transform_prompt | chat | StrOutputParser() | retriever,
    ).with_config(run_name="chat_retriever_chain")


    # Create conversational retrieval chain (parent of document chain and query transforming retriever chain. higher-level)
    conversational_retrieval_chain = RunnablePassthrough.assign(  # 1st Assignment
        context=query_transforming_retriever_chain,
    ).assign(                                                     # 2nd Assignment
        answer=document_chain,
    )
    """
    RunnablePassthrough class:
    - A LangChain utility class that helps create sequential processing chains
    - Allows you to "pass through" data while adding or transforming fields
    - Each .assign() adds a new field to the output dictionary

    1st Assignment (context): context=query_transforming_retriever_chain
    - Result stored in 'context' key of output dictionary

    2nd Assignment (answer): answer=document_chain
    - Takes the current state (including context from first assignment)
    - Passes it through the document_chain
    - Stores result in 'answer' key of output dictionary
    """

    return conversational_retrieval_chain


# Initialize components and chain at module level
_retriever, _chat = initialize_components()
_chain = create_chain(_retriever, _chat)


def rag_respond_chat_iface(message, history):
    """
    Process a message (question) and the chat history, and return an answer using the conversational retrieval chain
    :param message: str, the latest message provided by Gradio
    :param history: a list of dictionaries, the chat history passed by Gradio
                    [{'role': 'user', 'content': 'who is jane eyre?'},
                    {'role': 'assistant', 'content': 'Jane Eyre is...'}]
    :return: str, the generated answer extracted from the conversational retrieval chain's response
    """
    # Transform message and history into Langcahin's message classes
    if not history:  # Only one message in the chat
        messages = [HumanMessage(content=message)]
    else:
        messages = [
            HumanMessage(content=_.get('content')) if _.get('role') == 'user'
            else AIMessage(content=_.get('content'))
            for _ in history
        ]
        messages.append(HumanMessage(content=message))

    # Invoke the retrieval chain with the messages
    response = _chain.invoke({
        "messages": messages,
    })

    return response['answer']

