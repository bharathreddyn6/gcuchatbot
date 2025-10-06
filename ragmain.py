"""
Advanced RAG College Information Chatbot - FastAPI Backend
Features: Smart chunking, hybrid search, re-ranking, conversation memory
"""

from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional, Dict, Any
import os
import json
from datetime import datetime
import uuid

# Core RAG components - use the implementations available in ragpipeline.py
from ragpipeline import (
    initialize_vector_store,
    DocumentIngestionPipeline,
    AdvancedRAGRetriever,
    ConversationalRAGChain,
)

# Initialize FastAPI
app = FastAPI(title="College Chatbot API", version="1.0.0")

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize components
# Initialize vector store and pipeline components
# Persist directory matches the repo chroma_db folder
vector_store = initialize_vector_store(persist_directory="./chroma_db")
doc_processor = DocumentIngestionPipeline(vector_store)
rag_retriever = AdvancedRAGRetriever(vector_store)
conversational_chain = ConversationalRAGChain(rag_retriever)

# Storage for feedback
feedback_store = []


# Request/Response models
class ChatRequest(BaseModel):
    query: str
    session_id: str


class ChatResponse(BaseModel):
    answer: str
    sources: List[str]
    query_rewritten: Optional[str] = None


class FeedbackRequest(BaseModel):
    message_id: str
    feedback: str  # 'positive' or 'negative'
    session_id: str


class UploadResponse(BaseModel):
    uploaded_files: List[str]
    chunks_created: int


@app.get("/")
async def root():
    return {
        "message": "College Information Chatbot API",
        "version": "1.0.0",
        "endpoints": {
            "POST /upload": "Upload PDFs/CSVs",
            "POST /chat": "Chat with the bot",
            "POST /feedback": "Submit feedback"
        }
    }


@app.post("/upload", response_model=UploadResponse)
async def upload_files(files: List[UploadFile] = File(...)):
    """
    Upload and process documents (PDFs, CSVs)
    - Extracts text and tables
    - Performs smart chunking
    - Generates embeddings
    - Stores in vector database
    """
    try:
        uploaded_files = []
        total_chunks = 0

        # Create temp directory
        os.makedirs("temp_uploads", exist_ok=True)

        for file in files:
            if not file.filename.endswith(('.pdf', '.csv')):
                raise HTTPException(
                    status_code=400,
                    detail=
                    f"Unsupported file type: {file.filename}. Only PDF and CSV allowed."
                )

            # Save file temporarily
            file_path = f"temp_uploads/{file.filename}"
            with open(file_path, "wb") as f:
                f.write(await file.read())

                # Process document (ingest and add to vector store)
                # The ingestion pipeline returns a list of Document objects
                chunks = doc_processor.ingest_file(file_path, file.filename)

            uploaded_files.append(file.filename)
            total_chunks += len(chunks)

            # Clean up temp file
            os.remove(file_path)

        return UploadResponse(uploaded_files=uploaded_files,
                              chunks_created=total_chunks)

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    """
    Main chat endpoint with Advanced RAG pipeline:
    1. Query rewriting with conversation context
    2. Hybrid retrieval (vector + BM25)
    3. Cross-encoder re-ranking
    4. Context compression
    5. Answer generation with citations
    """
    try:
        # Delegate processing to conversational RAG chain
        result = conversational_chain.process_query(request.query,
                                                    request.session_id)

        return ChatResponse(answer=result['answer'],
                            sources=[
                                d.metadata.get('source', 'unknown')
                                for d in result.get('source_documents', [])
                            ],
                            query_rewritten=result.get('rewritten_query'))

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/feedback")
async def submit_feedback(request: FeedbackRequest):
    """
    Collect user feedback for continuous improvement
    """
    feedback_entry = {
        "message_id": request.message_id,
        "feedback": request.feedback,
        "session_id": request.session_id,
        "timestamp": datetime.now().isoformat()
    }
    feedback_store.append(feedback_entry)

    # Save to file
    os.makedirs("feedback", exist_ok=True)
    with open("feedback/feedback.jsonl", "a") as f:
        f.write(json.dumps(feedback_entry) + "\n")

    return {"status": "success", "message": "Feedback recorded"}


@app.get("/stats")
async def get_stats():
    """
    Get system statistics
    """
    # Try to surface some basic stats from the vector store and conversation chain
    try:
        # Chroma store exposes a get() method returning documents
        docs = vector_store.get().get('documents', [])
        total_documents = len(docs)
    except Exception:
        total_documents = 0

    try:
        active_sessions = conversational_chain.get_active_sessions()
    except Exception:
        active_sessions = 0

    return {
        "total_documents": total_documents,
        "total_chunks": total_documents,
        "active_sessions": active_sessions,
        "feedback_received": len(feedback_store)
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
