import os
import gradio as gr
from utils.env_setup import setup_environment
from rag_respond_chat_iface import rag_respond_chat_iface


# Set up environment at startup
setup_environment()
EMBEDDING_DEPLOYMENT_NAME = os.getenv('EMBEDDING_DEPLOYMENT_NAME')
LLM_DEPLOYMENT_NAME = os.getenv('LLM_DEPLOYMENT_NAME')

# Create Gradio chat interface
iface = gr.ChatInterface(
    fn=rag_respond_chat_iface,
    type="messages",  # Passes the history as a list of dictionaries with openai-style "role" and "content" keys to fn.
                      # [{'role': 'user', 'content': 'who is jane eyre?'},
                      # {'role': 'assistant', 'content': 'Jane Eyre is...'}]
    examples=["Who is Jane Eyre?"],
    example_labels=["Ask: Who is Jane Eyre?"],
    title="Jane Eyre RAG Chat",
    description=f"Ask questions about Jane Eyre (by Charlotte Brontë, 1847) and get answers based on the novel's content."
                f"<br />Using embedding model: {EMBEDDING_DEPLOYMENT_NAME}"
                f"<br />Using LLM model: {LLM_DEPLOYMENT_NAME}"
)


if __name__ == "__main__":
    iface.launch()

