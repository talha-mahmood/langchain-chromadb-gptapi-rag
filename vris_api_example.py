"""
VRIS Frontend Integration Example
Demonstrates how to integrate VRIS with a web frontend (FastAPI example)
"""

from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from typing import List, Dict, Any
from pathlib import Path
import shutil
import os
from datetime import datetime, timedelta
import tempfile
import asyncio
from vris_rag_system import VRISRAGSystem

app = FastAPI(title="VRIS API", version="1.0.0")

# CORS middleware for frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure for your frontend domain
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Configuration
SYSTEM_DOCS_FOLDER = "./system-doc"
UPLOAD_TEMP_FOLDER = "./temp_uploads"
CHROMA_DB_DIR = "./chroma_db"
FILE_RETENTION_HOURS = 72

# Ensure folders exist
os.makedirs(UPLOAD_TEMP_FOLDER, exist_ok=True)

# Initialize VRIS with pre-built system vectorstore
# This happens once at startup - system docs are already indexed
print("Initializing VRIS system...")
vris_base = VRISRAGSystem(
    system_docs_folder=SYSTEM_DOCS_FOLDER,
    veteran_docs_folder="",  # Not used in API mode
    persist_directory=CHROMA_DB_DIR,
    model_name="gpt-4"
)
# Pre-load system vectorstore only
vris_base.create_system_vectorstore(force_reload=False)
print("✓ System vectorstore ready")


def schedule_file_deletion(file_paths: List[str], hours: int = 72):
    """
    Schedule file deletion after specified hours
    In production, use Celery or similar task queue
    """
    async def delete_after_delay():
        await asyncio.sleep(hours * 3600)
        for file_path in file_paths:
            try:
                if os.path.exists(file_path):
                    os.remove(file_path)
                    print(f"Deleted file after {hours} hours: {file_path}")
            except Exception as e:
                print(f"Error deleting file {file_path}: {e}")
    
    # Schedule deletion
    asyncio.create_task(delete_after_delay())


@app.post("/api/vris/upload-documents")
async def upload_documents(files: List[UploadFile] = File(...)) -> Dict[str, Any]:
    """
    Upload veteran documents for processing
    Returns session ID for subsequent analysis requests
    """
    try:
        # Create session-specific folder
        session_id = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        session_folder = Path(UPLOAD_TEMP_FOLDER) / session_id
        session_folder.mkdir(parents=True, exist_ok=True)
        
        uploaded_files = []
        file_info = []
        
        for file in files:
            # Validate file type
            file_ext = Path(file.filename).suffix.lower()
            if file_ext not in ['.pdf', '.docx', '.txt']:
                continue
            
            # Save uploaded file
            file_path = session_folder / file.filename
            with open(file_path, "wb") as buffer:
                shutil.copyfileobj(file.file, buffer)
            
            uploaded_files.append(str(file_path))
            file_info.append({
                "filename": file.filename,
                "size": os.path.getsize(file_path),
                "type": file_ext[1:]
            })
        
        if not uploaded_files:
            raise HTTPException(status_code=400, detail="No valid files uploaded")
        
        # Schedule deletion after 72 hours
        schedule_file_deletion(uploaded_files, hours=FILE_RETENTION_HOURS)
        
        return {
            "success": True,
            "session_id": session_id,
            "files_uploaded": len(uploaded_files),
            "files": file_info,
            "expiry_hours": FILE_RETENTION_HOURS,
            "message": f"Files uploaded successfully. Will be deleted in {FILE_RETENTION_HOURS} hours."
        }
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/vris/analyze/free-snapshot")
async def free_snapshot(session_id: str) -> Dict[str, Any]:
    """
    Generate Free Rating Snapshot for uploaded documents
    No detailed CFR analysis - high-level overview only
    """
    try:
        session_folder = Path(UPLOAD_TEMP_FOLDER) / session_id
        if not session_folder.exists():
            raise HTTPException(status_code=404, detail="Session not found or expired")
        
        # Get uploaded file paths
        file_paths = [str(f) for f in session_folder.glob("*") if f.is_file()]
        if not file_paths:
            raise HTTPException(status_code=400, detail="No files found in session")
        
        # Create VRIS instance with veteran documents
        vris = VRISRAGSystem(
            system_docs_folder=SYSTEM_DOCS_FOLDER,
            veteran_docs_folder="",  # Not used
            persist_directory=CHROMA_DB_DIR,
            model_name="gpt-4"
        )
        
        # Use pre-loaded system vectorstore
        vris.system_vectorstore = vris_base.system_vectorstore
        
        # Process veteran documents (in-memory only, not persisted)
        vris.process_veteran_documents_from_upload(file_paths)
        
        # Generate free snapshot
        result = vris.generate_free_snapshot()
        
        return {
            "success": True,
            "session_id": session_id,
            "analysis_type": "Free Rating Snapshot",
            "data": result,
            "timestamp": datetime.now().isoformat()
        }
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/vris/analyze/initial-vre")
async def initial_vre(session_id: str) -> Dict[str, Any]:
    """
    Generate Initial VRE for first-time filers
    Full CFR analysis with rating recommendations
    """
    try:
        session_folder = Path(UPLOAD_TEMP_FOLDER) / session_id
        if not session_folder.exists():
            raise HTTPException(status_code=404, detail="Session not found or expired")
        
        file_paths = [str(f) for f in session_folder.glob("*") if f.is_file()]
        if not file_paths:
            raise HTTPException(status_code=400, detail="No files found in session")
        
        # Create VRIS instance
        vris = VRISRAGSystem(
            system_docs_folder=SYSTEM_DOCS_FOLDER,
            veteran_docs_folder="",
            persist_directory=CHROMA_DB_DIR,
            model_name="gpt-4"
        )
        
        vris.system_vectorstore = vris_base.system_vectorstore
        vris.process_veteran_documents_from_upload(file_paths)
        
        # Generate Initial VRE
        result = vris.generate_initial_vre()
        
        return {
            "success": True,
            "session_id": session_id,
            "analysis_type": "Initial VRE (First-Time Filer)",
            "data": result,
            "timestamp": datetime.now().isoformat()
        }
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/vris/analyze/second-look")
async def second_look_vre(session_id: str) -> Dict[str, Any]:
    """
    Generate Second Look VRE for already-rated veterans
    Identifies underratings, missed conditions, secondary opportunities
    """
    try:
        session_folder = Path(UPLOAD_TEMP_FOLDER) / session_id
        if not session_folder.exists():
            raise HTTPException(status_code=404, detail="Session not found or expired")
        
        file_paths = [str(f) for f in session_folder.glob("*") if f.is_file()]
        if not file_paths:
            raise HTTPException(status_code=400, detail="No files found in session")
        
        # Create VRIS instance
        vris = VRISRAGSystem(
            system_docs_folder=SYSTEM_DOCS_FOLDER,
            veteran_docs_folder="",
            persist_directory=CHROMA_DB_DIR,
            model_name="gpt-4"
        )
        
        vris.system_vectorstore = vris_base.system_vectorstore
        vris.process_veteran_documents_from_upload(file_paths)
        
        # Generate Second Look VRE
        result = vris.generate_second_look_vre()
        
        return {
            "success": True,
            "session_id": session_id,
            "analysis_type": "Second Look VRE (Already-Rated)",
            "data": result,
            "timestamp": datetime.now().isoformat()
        }
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/vris/query/custom")
async def custom_query(
    session_id: str,
    query: str,
    pipeline: str = "vris-b"  # "vris-a" or "vris-b"
) -> Dict[str, Any]:
    """
    Custom query on veteran documents
    pipeline: "vris-a" for extraction, "vris-b" for reasoning
    """
    try:
        session_folder = Path(UPLOAD_TEMP_FOLDER) / session_id
        if not session_folder.exists():
            raise HTTPException(status_code=404, detail="Session not found or expired")
        
        file_paths = [str(f) for f in session_folder.glob("*") if f.is_file()]
        if not file_paths:
            raise HTTPException(status_code=400, detail="No files found in session")
        
        # Create VRIS instance
        vris = VRISRAGSystem(
            system_docs_folder=SYSTEM_DOCS_FOLDER,
            veteran_docs_folder="",
            persist_directory=CHROMA_DB_DIR,
            model_name="gpt-4"
        )
        
        vris.system_vectorstore = vris_base.system_vectorstore
        vris.process_veteran_documents_from_upload(file_paths)
        
        # Execute query
        if pipeline.lower() == "vris-a":
            result = vris.vris_a_extract(query)
        else:
            result = vris.vris_b_analyze(query)
        
        return {
            "success": True,
            "session_id": session_id,
            "pipeline": pipeline.upper(),
            "query": query,
            "result": result,
            "timestamp": datetime.now().isoformat()
        }
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.delete("/api/vris/session/{session_id}")
async def delete_session(session_id: str) -> Dict[str, Any]:
    """
    Manually delete session files before 72-hour expiry
    """
    try:
        session_folder = Path(UPLOAD_TEMP_FOLDER) / session_id
        if not session_folder.exists():
            raise HTTPException(status_code=404, detail="Session not found")
        
        # Delete all files in session
        shutil.rmtree(session_folder)
        
        return {
            "success": True,
            "message": f"Session {session_id} deleted successfully"
        }
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/vris/health")
async def health_check() -> Dict[str, Any]:
    """
    Health check endpoint
    """
    return {
        "status": "healthy",
        "system_vectorstore_ready": vris_base.system_vectorstore is not None,
        "timestamp": datetime.now().isoformat()
    }


if __name__ == "__main__":
    import uvicorn
    print("\n" + "="*70)
    print("VRIS API Server Starting")
    print("="*70)
    print("System vectorstore pre-loaded and ready")
    print("Veteran documents will be processed on-demand")
    print("Files auto-delete after 72 hours")
    print("="*70 + "\n")
    
    uvicorn.run(app, host="0.0.0.0", port=8000)
