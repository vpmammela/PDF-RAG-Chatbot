import gradio as gr
from .chat import upload_document, chat_with_docs

def create_ui() -> gr.Blocks:
    """Create and return the Gradio interface"""
    with gr.Blocks(title="RAG Chatbot (OpenAI + ChromaDB)") as demo:
        gr.Markdown("# RAG Chatbot\nUpload documents and ask questions about them!")
        
        with gr.Tab("Upload Documents"):
            file_input = gr.File(
                label="Upload PDF or TXT files",
                file_types=[".pdf", ".txt"],
                type="filepath"
            )
            upload_button = gr.Button("Process Document")
            upload_output = gr.Textbox(label="Upload Status")
            
            upload_button.click(
                fn=upload_document,
                inputs=[file_input],
                outputs=[upload_output]
            )
        
        with gr.Tab("Chat"):
            chatbot = gr.Chatbot(label="Chat History")
            msg = gr.Textbox(label="Ask a question about your documents")
            clear = gr.Button("Clear")

            def respond(message, history):
                bot_message = chat_with_docs(message)
                history.append((message, bot_message))
                return "", history

            msg.submit(respond, [msg, chatbot], [msg, chatbot])
            clear.click(lambda: None, None, chatbot, queue=False)
    
    return demo 