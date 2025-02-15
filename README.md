# RAG Chatbot

A Retrieval-Augmented Generation (RAG) chatbot that can answer questions about uploaded documents using OpenAI's GPT-4o-mini and ChromaDB for vector storage.

## Project Structure

## Setup

1. Clone the repository
2. Create a virtual environment: `python -m venv venv`
3. Activate the virtual environment:
   - Windows: `venv\Scripts\activate`
   - Unix/MacOS: `source venv/bin/activate`
4. Install dependencies: `pip install -r requirements.txt`
5. Copy `.env.example` to `.env` and add your OpenAI API key
6. Run the application: `python Main.py`

## Features

- Supports PDF and TXT file uploads
- Document chunking for better context handling
- Vector similarity search using ChromaDB
- Conversation history tracking
- Source document attribution in responses

## Usage

1. Upload one or more PDF or TXT files
2. Ask questions about the content of the documents
3. The chatbot will provide answers based on the document content with source attribution