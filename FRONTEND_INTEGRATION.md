# VRIS Frontend Integration Guide

## Architecture Overview

### System Documents (Pre-Built)
- ✅ **Stored in ChromaDB** at initialization
- ✅ **Persisted** across all sessions
- ✅ **Loaded once** at server startup
- Contains: VRIS docs, CFR rules, rating criteria

### Veteran Documents (Dynamic)
- ✅ **Uploaded by users** via frontend
- ✅ **Processed in-memory** (not persisted to ChromaDB)
- ✅ **Auto-deleted** after 72 hours
- ✅ **Session-based** (each veteran gets unique session)

## How It Works

### 1. Server Startup
```python
# System vectorstore is built ONCE and persisted
vris = VRISRAGSystem(...)
vris.create_system_vectorstore()  # Loads from disk if exists
```

### 2. User Upload Flow
```
Frontend → Upload PDFs → API /upload-documents → Session ID returned
```

### 3. Analysis Flow
```
Frontend → Request analysis with Session ID → 
API loads veteran docs from session folder →
Process in-memory (no persistence) →
Return analysis results
```

### 4. Cleanup
```
After 72 hours → Auto-delete veteran files
OR
Immediate → DELETE /session/{id}
```

## API Endpoints

### Upload Documents
```bash
POST /api/vris/upload-documents
Content-Type: multipart/form-data

# Returns:
{
  "session_id": "20260201_143022_123456",
  "files_uploaded": 4,
  "expiry_hours": 72
}
```

### Free Rating Snapshot
```bash
POST /api/vris/analyze/free-snapshot
Body: {"session_id": "20260201_143022_123456"}

# Returns high-level analysis without detailed CFR
```

### Initial VRE (First-Time Filer)
```bash
POST /api/vris/analyze/initial-vre
Body: {"session_id": "20260201_143022_123456"}

# Returns full analysis with CFR citations
```

### Second Look VRE (Already-Rated)
```bash
POST /api/vris/analyze/second-look
Body: {"session_id": "20260201_143022_123456"}

# Returns underratings, missed conditions, secondary opportunities
```

### Custom Query
```bash
POST /api/vris/query/custom
Body: {
  "session_id": "20260201_143022_123456",
  "query": "What evidence supports a higher PTSD rating?",
  "pipeline": "vris-b"
}
```

### Delete Session
```bash
DELETE /api/vris/session/{session_id}
```

## Frontend Integration Example (React)

```javascript
// 1. Upload Documents
const uploadDocuments = async (files) => {
  const formData = new FormData();
  files.forEach(file => formData.append('files', file));
  
  const response = await fetch('/api/vris/upload-documents', {
    method: 'POST',
    body: formData
  });
  
  const data = await response.json();
  return data.session_id;
};

// 2. Get Free Snapshot
const getFreeSnapshot = async (sessionId) => {
  const response = await fetch('/api/vris/analyze/free-snapshot', {
    method: 'POST',
    headers: {'Content-Type': 'application/json'},
    body: JSON.stringify({session_id: sessionId})
  });
  
  return await response.json();
};

// 3. Get Second Look VRE
const getSecondLookVRE = async (sessionId) => {
  const response = await fetch('/api/vris/analyze/second-look', {
    method: 'POST',
    headers: {'Content-Type': 'application/json'},
    body: JSON.stringify({session_id: sessionId})
  });
  
  return await response.json();
};

// Complete Flow
const analyzeVeteranDocuments = async () => {
  try {
    // Upload
    const sessionId = await uploadDocuments(selectedFiles);
    
    // Analyze
    const snapshot = await getFreeSnapshot(sessionId);
    
    if (userWantsPaidAnalysis) {
      const fullAnalysis = await getSecondLookVRE(sessionId);
      displayResults(fullAnalysis);
    }
    
  } catch (error) {
    console.error('Analysis failed:', error);
  }
};
```

## Key Differences from Original

### Before (Original):
```python
# Both system AND veteran docs persisted in ChromaDB
vris.initialize(force_reload=False)  
# Would load veteran docs from disk
```

### After (Current):
```python
# System docs: Pre-built and persisted
vris.create_system_vectorstore()  # Loads from disk

# Veteran docs: Fresh upload, in-memory only
vris.process_veteran_documents_from_upload(file_paths)  # No persistence
```

## Benefits

✅ **Fast System Startup**: System docs loaded once, instant for subsequent requests
✅ **Privacy Compliant**: Veteran docs never persisted, deleted after 72 hours
✅ **Scalable**: Each veteran session is isolated
✅ **Efficient**: System knowledge base shared across all users
✅ **Secure**: No cross-contamination between veteran sessions

## Running the API

### Install Additional Dependencies
```bash
pip install fastapi uvicorn python-multipart
```

### Start the Server
```bash
python vris_api_example.py
```

Server runs on `http://localhost:8000`

### API Documentation
Visit `http://localhost:8000/docs` for interactive Swagger UI

## Production Considerations

1. **File Deletion**: Use Celery/Redis for scheduled deletion instead of asyncio
2. **Session Management**: Use Redis for session storage
3. **Rate Limiting**: Add rate limits per user/IP
4. **Authentication**: Add JWT authentication
5. **Encryption**: Encrypt uploaded files at rest
6. **Load Balancing**: Use multiple workers for concurrent requests
7. **Monitoring**: Add logging and metrics
8. **HIPAA Compliance**: Use Azure OpenAI with BAA

## Testing

```bash
# Health check
curl http://localhost:8000/api/vris/health

# Upload documents
curl -X POST http://localhost:8000/api/vris/upload-documents \
  -F "files=@decision_letter.pdf" \
  -F "files=@cp_exam.pdf"

# Analyze (use session_id from upload response)
curl -X POST http://localhost:8000/api/vris/analyze/second-look \
  -H "Content-Type: application/json" \
  -d '{"session_id": "20260201_143022_123456"}'
```
