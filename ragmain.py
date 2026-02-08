"""
Advanced RAG College Information Chatbot - FastAPI Backend
Features: Smart chunking, hybrid search, re-ranking, conversation memory
"""

from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import HTMLResponse
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
from preload_documents import load_document_cache
from gcu_integration import GCUPortal

# Module handlers for Library, Study Plan, Complaints
from module_handlers import route_module_query
from module_config import config_manager

# HARDCODED CREDENTIALS FOR DEMO
# In production, ask user or use env vars
GCU_USERNAME = "22btcs128@gcu.edu.in"
GCU_PASSWORD = "Soz38610"

# Initialize FastAPI
app = FastAPI(title="College Chatbot API", version="1.0.0")

# Global Portal Instance
gcu_portal = GCUPortal()

@app.on_event("shutdown")
def shutdown_event():
    print("Shutting down... closing portal driver.")
    gcu_portal.close()

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

try:
    preload_content = load_document_cache("college_data")
    doc_processor = DocumentIngestionPipeline(
        vector_store,
        preloaded_csvs=preload_content.csv_data,
        preloaded_pdfs=preload_content.pdf_pages)
except Exception as exc:
    # Fallback gracefully if preloading fails (e.g., missing deps)
    print(f"[WARN] Skipping document cache preload: {exc}")
    doc_processor = DocumentIngestionPipeline(vector_store)
rag_retriever = AdvancedRAGRetriever(vector_store)
conversational_chain = ConversationalRAGChain(rag_retriever)

# Storage for feedback
feedback_store = []


# Request/Response models
class ChatRequest(BaseModel):
    query: str
    session_id: str
    context_mode: Optional[str] = None


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
    message: Optional[str] = None


@app.get("/")
async def root():
    return {
        "message": "College Information Chatbot API",
        "version": "1.0.0",
        "endpoints": {
            "POST /upload": "Upload PDFs/CSVs",
            "POST /chat": "Chat with the bot",
            "GET /voice": "Voice Interface",
            "POST /feedback": "Submit feedback"
        }
    }


@app.get("/voice", response_class=HTMLResponse)
async def voice_interface():
    """Serve the voice chat interface HTML"""
    if os.path.exists("voice_chat.html"):
        with open("voice_chat.html", "r", encoding="utf-8") as f:
            return f.read()
    return "Voice chat interface not found. Please ensure 'voice_chat.html' is in the root directory."


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
                      chunks_created=total_chunks,
                      message=f"Successfully processed {len(uploaded_files)} files and created {total_chunks} chunks.")

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


def check_portal_data(query_text, force_check=False):
    """
    Checks if the query is about student specific data (attendance, results, etc.)
    If so, logs in to the portal and fetches data.
    """
    keywords = ["attendance", "result", "marks", "grade", "fee", "dashboard", "academic", "info"]
    
    should_check = force_check or (any(k in query_text.lower() for k in keywords) and "my" in query_text.lower())
    
    if should_check:
        try:
            print("Detected request for student data. Checking portal session...")
            # Use global instance
            if gcu_portal.login(GCU_USERNAME, GCU_PASSWORD):
                data = gcu_portal.get_student_data()
                # Do NOT close here, keep it open for next time
                return data
            else:
                return "I tried to log in to your portal but failed. Please check the credentials."
        except Exception as e:
            print(f"Portal check error: {e}")
            return "I encountered an error while trying to access the college portal."
    return None


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
    print(f"Received query: {request.query} | Session: {request.session_id} | Mode: {request.context_mode}")
    
    try:
        # 0. Check for Personal Data Request (Portal Integration)
        # Check for Academic Mode
        if request.context_mode == "Academic Info":
            # In Academic Mode, route to specific GRMS pages based on query
            query_lower = request.query.lower()
            
            # Ensure login first
            if not gcu_portal.login(GCU_USERNAME, GCU_PASSWORD):
                return ChatResponse(answer="Failed to log in to GRMS portal. Please check credentials.", sources=["GCU Portal"])
            
            # Route based on keywords
            if any(k in query_lower for k in ["event", "upcoming", "fest", "workshop", "seminar", "colloquium", "programme", "program"]):
                data = gcu_portal.get_events()
                return ChatResponse(answer=data, sources=["GCU Portal - Events Page"])
            
            elif any(k in query_lower for k in ["attendance", "present", "absent", "class"]):
                data = gcu_portal.get_attendance()
                return ChatResponse(answer=data, sources=["GCU Portal - Attendance Page"])
            
            else:
                # General query - use dashboard
                data = gcu_portal.get_student_data()
                return ChatResponse(answer=data, sources=["GCU Portal - Dashboard"])
        
        # Check for Library Mode - Route to module handler
        if request.context_mode == "Library":
            result = await route_module_query("Library", request.query, request.session_id)
            return ChatResponse(answer=result["answer"], sources=result.get("sources", ["Library"]))
        
        # Check for Study Plan Mode - Route to module handler
        if request.context_mode == "Study Plan":
            result = await route_module_query("Study Plan", request.query, request.session_id)
            return ChatResponse(answer=result["answer"], sources=result.get("sources", ["Study Plan"]))
        
        # Check for Complaints Mode - Route to module handler
        if request.context_mode == "Complaints":
            result = await route_module_query("Complaints", request.query, request.session_id)
            return ChatResponse(answer=result["answer"], sources=result.get("sources", ["Complaints"]))
        
        # Check for specific "check my..." queries (Legacy/Global check)
        # This acts as a fallback if no specific context mode was active, or if the context mode didn't yield a result
        if "check my" in request.query.lower():
            portal_data = check_portal_data(request.query)
            if portal_data:
                return ChatResponse(answer=portal_data, sources=["GCU Portal"])
        
        # Original portal check (if not handled by context mode or "check my" keyword)
        portal_response = check_portal_data(request.query)
        if portal_response:
            return ChatResponse(answer=portal_response,
                                sources=["GCU Student Portal (Live)"],
                                query_rewritten=request.query)

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

    # Feedback breakdown
    positive_feedback = sum(1 for f in feedback_store if f.get('feedback') == 'positive')
    negative_feedback = sum(1 for f in feedback_store if f.get('feedback') == 'negative')

    return {
        # Knowledge base
        "total_documents": total_documents,
        "total_chunks": total_documents,
        "vector_store_type": "Chroma",
        "embedding_model": "all-MiniLM-L6-v2",
        # Conversations
        "active_sessions": active_sessions,
        "active_conversations": active_sessions,
        # Feedback
        "feedback_received": len(feedback_store),
        "positive_feedback": positive_feedback,
        "negative_feedback": negative_feedback
    }

@app.get("/health")
async def health():
    """Basic health check endpoint"""
    return {"status": "ok", "version": "1.0.0"}

@app.delete("/clear_memory/{session_id}")
async def clear_memory(session_id: str):
    """Clear conversation memory for given session"""
    try:
        conversational_chain.clear_memory(session_id)
        return {"status": "success", "message": f"Cleared memory for session {session_id}"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
