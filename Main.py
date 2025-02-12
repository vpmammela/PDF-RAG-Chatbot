import os
import chromadb
import gradio as gr
from openai import OpenAI
from dotenv import load_dotenv
from pathlib import Path
import PyPDF2
import json

# Load environment variables
load_dotenv()
client = OpenAI(api_key=os.getenv('OPENAI_API_KEY'))

# Initialize ChromaDB
PERSIST_DIR = "./chroma_db"
chroma_client = chromadb.PersistentClient(path=PERSIST_DIR)
collection = chroma_client.get_or_create_collection("documents")

def extract_text_from_pdf(file_path):
    """Extract text from PDF file"""
    with open(file_path, 'rb') as file:
        reader = PyPDF2.PdfReader(file)
        text = ''
        for page in reader.pages:
            text += page.extract_text() + '\n'
    return text

def read_file(file_path):
    """Read content from a file (PDF or TXT)"""
    if file_path.endswith('.pdf'):
        return extract_text_from_pdf(file_path)
    else:
        with open(file_path, 'r', encoding='utf-8') as file:
            return file.read()

def get_embeddings(text):
    """Get embeddings from OpenAI API"""
    response = client.embeddings.create(
        model="text-embedding-3-small",
        input=text,
        dimensions=1536
    )
    return response.data[0].embedding

def process_document(file_path):
    """Process a document and add it to ChromaDB"""
    text = read_file(file_path)
    
    # Get embeddings for the document
    embeddings = get_embeddings(text)
    
    # Add to ChromaDB
    collection.add(
        embeddings=[embeddings],
        documents=[text],
        ids=[str(Path(file_path).stem)]
    )

def query_documents(question):
    """Query documents using embeddings similarity"""
    # Get embeddings for the question
    question_embedding = get_embeddings(question)
    
    # Query ChromaDB
    results = collection.query(
        query_embeddings=[question_embedding],
        n_results=3
    )
    
    # Prepare context from similar documents
    context = "\n".join(results['documents'][0])
    
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
    
    return response.choices[0].message.content

def answer(message, history):
    """Main function to handle chat interface"""
    try:
        # Process any new files
        if "files" in message:
            for file in message["files"]:
                process_document(file)
        
        # Process files from history
        for msg in history:
            if msg['role'] == "user" and isinstance(msg['content'], tuple):
                process_document(msg['content'][0])
        
        # Query the documents
        response = query_documents(message["text"])
        return response
        
    except Exception as e:
        return f"Error: {str(e)}"

# Create Gradio interface
demo = gr.ChatInterface(
    answer,
    type="messages",
    title="RAG Chatbot (OpenAI + ChromaDB)",
    description="Upload PDF or TXT files and ask questions about them!",
    textbox=gr.MultimodalTextbox(file_types=[".pdf", ".txt"]),
    multimodal=True
)

if __name__ == "__main__":
    demo.launch()