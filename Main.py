import os
import chromadb
import gradio as gr
from openai import OpenAI
from dotenv import load_dotenv
from pathlib import Path
import PyPDF2
import json
import logging
from typing import List, Dict, Union, Any
from textwrap import wrap

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()
client = OpenAI(api_key=os.getenv('OPENAI_API_KEY'))

# Initialize ChromaDB
PERSIST_DIR = "./chroma_db"
chroma_client = chromadb.PersistentClient(path=PERSIST_DIR)
collection = chroma_client.get_or_create_collection("documents")

# Constants
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200
MAX_RETRIES = 3

def chunk_text(text: str, chunk_size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> List[str]:
    """Split text into overlapping chunks"""
    chunks = []
    text_length = len(text)
    start = 0
    
    while start < text_length:
        end = start + chunk_size
        chunk = text[start:end]
        chunks.append(chunk)
        start = end - overlap
        
    return chunks

def extract_text_from_pdf(file_path: str) -> str:
    """Extract text from PDF file"""
    try:
        with open(file_path, 'rb') as file:
            reader = PyPDF2.PdfReader(file)
            text = ''
            for page in reader.pages:
                text += page.extract_text() + '\n'
        return text
    except Exception as e:
        logger.error(f"Error extracting text from PDF {file_path}: {str(e)}")
        raise

def read_file(file_path: str) -> str:
    """Read content from a file (PDF or TXT)"""
    try:
        if file_path.endswith('.pdf'):
            return extract_text_from_pdf(file_path)
        else:
            with open(file_path, 'r', encoding='utf-8') as file:
                return file.read()
    except Exception as e:
        logger.error(f"Error reading file {file_path}: {str(e)}")
        raise

def get_embeddings(text: str, retries: int = MAX_RETRIES) -> List[float]:
    """Get embeddings from OpenAI API with retry logic"""
    for attempt in range(retries):
        try:
            response = client.embeddings.create(
                model="text-embedding-3-small",
                input=text,
                dimensions=1536
            )
            return response.data[0].embedding
        except Exception as e:
            if attempt == retries - 1:
                logger.error(f"Failed to get embeddings after {retries} attempts: {str(e)}")
                raise
            logger.warning(f"Embedding attempt {attempt + 1} failed, retrying...")

def upload_document(file_path: str) -> str:
    """Upload and process a document"""
    try:
        logger.info(f"Starting to process document: {file_path}")
        text = read_file(file_path)
        chunks = chunk_text(text)
        
        logger.info(f"Document split into {len(chunks)} chunks")
        
        for i, chunk in enumerate(chunks):
            embeddings = get_embeddings(chunk)
            collection.add(
                embeddings=[embeddings],
                documents=[chunk],
                ids=[f"{Path(file_path).stem}_chunk_{i}"],
                metadatas=[{
                    "source": str(file_path),
                    "chunk_index": i,
                    "total_chunks": len(chunks)
                }]
            )
            logger.debug(f"Processed chunk {i+1}/{len(chunks)} for {file_path}")
            
        logger.info(f"✨ Document successfully processed and ready to use: {file_path}")
        return f"Successfully processed document: {Path(file_path).name}"
    except Exception as e:
        error_msg = f"Error processing document {file_path}: {str(e)}"
        logger.error(error_msg)
        return error_msg

def query_documents(question: str) -> str:
    """Query documents using embeddings similarity"""
    try:
        # Get embeddings for the question
        question_embedding = get_embeddings(question)
        
        # Query ChromaDB
        results = collection.query(
            query_embeddings=[question_embedding],
            n_results=3,
            include=["documents", "metadatas"]
        )
        
        # Prepare context from similar documents
        context = "\n".join(results['documents'][0])
        sources = [meta["source"] for meta in results["metadatas"][0]]
        
        # Prepare the message for GPT
        messages = [
            {"role": "system", "content": "You are a helpful assistant. Use the provided context to answer questions. If the answer cannot be found in the context, say so."},
            {"role": "user", "content": f"Context:\n{context}\n\nQuestion: {question}"}
        ]
        
        # Get response from GPT
        response = client.chat.completions.create(
            model="gpt-4o-mini", 
            messages=messages,
            temperature=0.7,
            max_tokens=500
        )
        
        answer = response.choices[0].message.content
        return f"{answer}\n\nSources: {', '.join(set(sources))}"
    
    except Exception as e:
        logger.error(f"Error querying documents: {str(e)}")
        raise

def chat_with_docs(message: str) -> str:
    """Query documents using the provided message"""
    try:
        # Check if any documents are loaded
        if collection.count() == 0:
            return "Please upload some documents first before asking questions."
            
        return query_documents(message)
    except Exception as e:
        error_msg = f"Error querying documents: {str(e)}"
        logger.error(error_msg)
        return error_msg

# Create Gradio interface with two tabs
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

if __name__ == "__main__":
    demo.launch()