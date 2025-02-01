import gradio as gr
from langchain.chains.question_answering.map_reduce_prompt import messages

from utils.env_setup import setup_environment
from rag_respond_chat_iface import rag_respond_chat_iface


# Set up environment at startup
setup_environment()


# Create Gradio chat interface
iface = gr.ChatInterface(
    fn=rag_respond_chat_iface,
    type="messages",  # Passes the history as a list of dictionaries with openai-style "role" and "content" keys to fn.
                      # [{'role': 'user', 'content': 'who is jane eyre?'},
                      # {'role': 'assistant', 'content': 'Jane Eyre is...'}]
    examples=["Who is Jane Eyre?"],
    example_labels=["Ask: Who is Jane Eyre?"],
    title="Jane Eyre RAG Chat",
    description="Ask questions about Jane Eyre (by Charlotte Brontë, 1847) and get answers based on the novel's content."
)


if __name__ == "__main__":
    iface.launch()

