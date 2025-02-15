import logging
from pathlib import Path
import chromadb
from .embeddings import client, get_embeddings

logger = logging.getLogger(__name__)

# Initialize ChromaDB
PERSIST_DIR = "./chroma_db"
chroma_client = chromadb.PersistentClient(path=PERSIST_DIR)
collection = chroma_client.get_or_create_collection("documents")

def upload_document(file_path: str) -> str:
    """Upload and process a document"""
    from .document_processor import read_file, chunk_text
    
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
        question_embedding = get_embeddings(question)
        
        results = collection.query(
            query_embeddings=[question_embedding],
            n_results=3,
            include=["documents", "metadatas"]
        )
        
        context = "\n".join(results['documents'][0])
        sources = [meta["source"] for meta in results["metadatas"][0]]
        
        messages = [
            {"role": "system", "content": "You are a helpful assistant. Use the provided context to answer questions. If the answer cannot be found in the context, say so."},
            {"role": "user", "content": f"Context:\n{context}\n\nQuestion: {question}"}
        ]
        
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
        if collection.count() == 0:
            return "Please upload some documents first before asking questions."
            
        return query_documents(message)
    except Exception as e:
        error_msg = f"Error querying documents: {str(e)}"
        logger.error(error_msg)
        return error_msg 