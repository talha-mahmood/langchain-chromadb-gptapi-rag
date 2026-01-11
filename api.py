"""
FastAPI application for PDF RAG system.
Allows external projects to query against local PDFs and user-uploaded PDFs.
"""

import os
import shutil
from pathlib import Path
from typing import Optional, List
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from dotenv import load_dotenv
import tempfile
import uvicorn

from pdf_rag_system import PDFRAGSystem

# Load environment variables
load_dotenv()

# Initialize FastAPI app
app = FastAPI(
    title="PDF RAG API",
    description="API for querying PDF documents using RAG (Retrieval-Augmented Generation)",
    version="1.0.0"
)

# Configuration
BASE_PDF_FOLDER = "./pdfs"  # Local RAG PDFs
USER_PDF_FOLDER = "./user_pdfs"  # Temporary storage for user PDFs
CHROMA_DB_BASE = "./chroma_db_base"  # Base vectorstore for local PDFs
CHROMA_DB_USER = "./chroma_db_user"  # Temporary vectorstore for user PDFs

# Create folders
os.makedirs(BASE_PDF_FOLDER, exist_ok=True)
os.makedirs(USER_PDF_FOLDER, exist_ok=True)

# Initialize base RAG system with local PDFs
base_rag = None


# Request/Response Models
class QueryRequest(BaseModel):
    question: str
    user_id: Optional[str] = None
    use_user_pdf: bool = False


class QueryResponse(BaseModel):
    question: str
    answer: str
    sources: List[dict]
    user_id: Optional[str] = None


class HealthResponse(BaseModel):
    status: str
    message: str
    base_documents: int


# Helper functions
def initialize_base_rag():
    """Initialize the base RAG system with local PDFs"""
    global base_rag
    try:
        base_rag = PDFRAGSystem(
            pdf_folder=BASE_PDF_FOLDER,
            persist_directory=CHROMA_DB_BASE
        )
        base_rag.initialize(force_reload=False)
        return True
    except Exception as e:
        print(f"Error initializing base RAG: {e}")
        return False


def get_user_pdf_path(user_id: str) -> str:
    """Get the path for user-specific PDF storage"""
    user_folder = os.path.join(USER_PDF_FOLDER, user_id)
    os.makedirs(user_folder, exist_ok=True)
    return user_folder


def get_user_chroma_path(user_id: str) -> str:
    """Get the path for user-specific ChromaDB"""
    return f"{CHROMA_DB_USER}_{user_id}"


def create_user_rag(user_id: str) -> PDFRAGSystem:
    """Create a RAG system that combines base PDFs and user PDFs"""
    user_pdf_path = get_user_pdf_path(user_id)
    user_chroma_path = get_user_chroma_path(user_id)
    
    # Create a temporary combined PDF folder
    temp_folder = tempfile.mkdtemp()
    
    # Copy base PDFs
    if os.path.exists(BASE_PDF_FOLDER):
        for pdf_file in Path(BASE_PDF_FOLDER).glob("*.pdf"):
            shutil.copy(pdf_file, temp_folder)
    
    # Copy user PDFs
    if os.path.exists(user_pdf_path):
        for pdf_file in Path(user_pdf_path).glob("*.pdf"):
            shutil.copy(pdf_file, temp_folder)
    
    # Initialize RAG with combined PDFs
    user_rag = PDFRAGSystem(
        pdf_folder=temp_folder,
        persist_directory=user_chroma_path
    )
    user_rag.initialize(force_reload=True)
    
    # Clean up temp folder
    shutil.rmtree(temp_folder)
    
    return user_rag


# API Endpoints

@app.on_event("startup")
async def startup_event():
    """Initialize the base RAG system on startup"""
    print("Initializing base RAG system...")
    success = initialize_base_rag()
    if success:
        print("Base RAG system initialized successfully")
    else:
        print("Warning: Base RAG system initialization failed")


@app.get("/", response_model=HealthResponse)
async def root():
    """Health check endpoint"""
    doc_count = 0
    if base_rag and base_rag.vectorstore:
        doc_count = base_rag.vectorstore._collection.count()
    
    return HealthResponse(
        status="healthy",
        message="PDF RAG API is running",
        base_documents=doc_count
    )


@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Detailed health check"""
    if not base_rag:
        raise HTTPException(status_code=503, detail="Base RAG system not initialized")
    
    doc_count = base_rag.vectorstore._collection.count() if base_rag.vectorstore else 0
    
    return HealthResponse(
        status="healthy",
        message="System operational",
        base_documents=doc_count
    )


@app.post("/upload-user-pdf")
async def upload_user_pdf(
    file: UploadFile = File(...),
    user_id: str = Form(...),
    user_name: Optional[str] = Form(None),
    user_email: Optional[str] = Form(None)
):
    """
    Upload a user-specific PDF file
    
    Args:
        file: PDF file to upload
        user_id: Unique identifier for the user
        user_name: Optional user name
        user_email: Optional user email
    """
    # Validate file type
    if not file.filename.endswith('.pdf'):
        raise HTTPException(status_code=400, detail="Only PDF files are allowed")
    
    try:
        # Get user folder
        user_folder = get_user_pdf_path(user_id)
        
        # Save the uploaded file
        file_path = os.path.join(user_folder, file.filename)
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        
        # Store user metadata (optional)
        metadata = {
            "user_id": user_id,
            "user_name": user_name,
            "user_email": user_email,
            "filename": file.filename
        }
        
        return JSONResponse(
            status_code=200,
            content={
                "message": "PDF uploaded successfully",
                "user_id": user_id,
                "filename": file.filename,
                "metadata": metadata
            }
        )
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error uploading file: {str(e)}")


@app.post("/query", response_model=QueryResponse)
async def query_pdf(request: QueryRequest):
    """
    Query the RAG system
    
    Args:
        request: QueryRequest containing question, user_id, and use_user_pdf flag
    
    Returns:
        QueryResponse with answer and sources
    """
    if not base_rag:
        raise HTTPException(status_code=503, detail="RAG system not initialized")
    
    try:
        # Determine which RAG system to use
        if request.use_user_pdf and request.user_id:
            # Check if user has uploaded PDFs
            user_pdf_path = get_user_pdf_path(request.user_id)
            user_pdfs = list(Path(user_pdf_path).glob("*.pdf"))
            
            if not user_pdfs:
                raise HTTPException(
                    status_code=404,
                    detail=f"No PDFs found for user {request.user_id}"
                )
            
            # Create user-specific RAG (combines base + user PDFs)
            print(f"Creating combined RAG for user {request.user_id}...")
            rag_system = create_user_rag(request.user_id)
        else:
            # Use base RAG only
            rag_system = base_rag
        
        # Query the system
        result = rag_system.query(request.question)
        
        # Format sources
        sources = []
        for doc in result.get('source_documents', []):
            sources.append({
                "source": Path(doc.metadata.get('source', 'Unknown')).name,
                "page": doc.metadata.get('page', 'Unknown'),
                "content_snippet": doc.page_content[:200] + "..." if len(doc.page_content) > 200 else doc.page_content
            })
        
        return QueryResponse(
            question=request.question,
            answer=result.get('answer', 'No answer generated'),
            sources=sources,
            user_id=request.user_id
        )
    
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing query: {str(e)}")


@app.post("/query-base-only")
async def query_base_only(question: str = Form(...)):
    """
    Query only the base RAG PDFs (simpler endpoint)
    
    Args:
        question: The question to ask
    """
    if not base_rag:
        raise HTTPException(status_code=503, detail="RAG system not initialized")
    
    try:
        result = base_rag.query(question)
        
        sources = []
        for doc in result.get('source_documents', []):
            sources.append({
                "source": Path(doc.metadata.get('source', 'Unknown')).name,
                "page": doc.metadata.get('page', 'Unknown')
            })
        
        return {
            "question": question,
            "answer": result.get('answer', 'No answer generated'),
            "sources": sources
        }
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing query: {str(e)}")


@app.delete("/delete-user-data/{user_id}")
async def delete_user_data(user_id: str):
    """
    Delete all data for a specific user
    
    Args:
        user_id: User identifier
    """
    try:
        # Delete user PDFs
        user_pdf_path = get_user_pdf_path(user_id)
        if os.path.exists(user_pdf_path):
            shutil.rmtree(user_pdf_path)
        
        # Delete user ChromaDB
        user_chroma_path = get_user_chroma_path(user_id)
        if os.path.exists(user_chroma_path):
            shutil.rmtree(user_chroma_path)
        
        return {
            "message": f"All data deleted for user {user_id}",
            "user_id": user_id
        }
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error deleting user data: {str(e)}")


@app.get("/list-base-pdfs")
async def list_base_pdfs():
    """List all PDFs in the base folder"""
    try:
        pdfs = [f.name for f in Path(BASE_PDF_FOLDER).glob("*.pdf")]
        return {
            "count": len(pdfs),
            "pdfs": pdfs
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error listing PDFs: {str(e)}")


@app.get("/list-user-pdfs/{user_id}")
async def list_user_pdfs(user_id: str):
    """List all PDFs for a specific user"""
    try:
        user_pdf_path = get_user_pdf_path(user_id)
        pdfs = [f.name for f in Path(user_pdf_path).glob("*.pdf")]
        return {
            "user_id": user_id,
            "count": len(pdfs),
            "pdfs": pdfs
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error listing user PDFs: {str(e)}")


if __name__ == "__main__":
    # Run the API server
    uvicorn.run(
        "api:app",
        host="0.0.0.0",
        port=8000,
        reload=True
    )
