# VRIS Testing Guide

## Prerequisites

✅ Python packages installed: `pip install -r vris_requirements.txt`
✅ OpenAI API key in `.env` file
✅ System documents in `system-doc/` folder (DOCX files already there)
✅ Sample veteran documents in `user_pdfs/1/` folder (TXT files already created)

## Test 1: Standalone VRIS System

### Run the Interactive System
```bash
python vris_rag_system.py
```

### What to Expect:
```
🚀 INITIALIZING VRIS™ RAG SYSTEM
Loading pre-built system vectorstore...
✓ Loaded system vectorstore with 42 chunks

Processing veteran documents (not persisted)...
Found 0 PDF, 0 DOCX, and 4 TXT file(s) in ./user_pdfs/1
✓ Veteran documents processed: 20 chunks in memory

✓ VRIS-A Extraction Pipeline initialized
✓ VRIS-B Reasoning Pipeline initialized

✅ VRIS™ RAG SYSTEM READY
```

### Test Each Workflow:

#### Option 1: Free Rating Snapshot
```
Select workflow (1-7): 1
```
**Expected:** High-level summary without detailed CFR citations
**Should identify:** Potential underratings and missed conditions

#### Option 2: Initial VRE
```
Select workflow (1-7): 2
```
**Expected:** Full analysis for first-time filers
**Should identify:** All claimable conditions with rating estimates

#### Option 3: Second Look VRE (⭐ Best for sample docs)
```
Select workflow (1-7): 3
```
**Expected Results:**
- ✅ PTSD underrating: 30% → 50% (90% confidence)
- ✅ Sleep Apnea missed: 0% → 50% (85% confidence)
- ✅ Depression secondary to PTSD (80% confidence)
- ✅ Potential combined rating increase: 60% → 80-90%

#### Option 4: Custom VRIS-A Extraction
```
Select workflow (1-7): 4

Enter VRIS-A extraction query:
Extract all service-connected conditions with their current ratings, diagnostic codes, 
and any conditions mentioned in medical records that are NOT currently rated.
```
**Expected:** Raw extracted data without reasoning

#### Option 5: Custom VRIS-B Analysis
```
Select workflow (1-7): 5

Enter VRIS-B reasoning query:
What specific 38 CFR criteria support increasing the PTSD rating from 30% to 50%?
```
**Expected:** Detailed CFR analysis with citations

#### Option 6: Interactive Query Mode
```
Select workflow (1-7): 6

Query: What secondary conditions should be claimed based on the veteran's evidence?
Query: Calculate the exact combined rating if all Phase 1 opportunities are granted.
Query: back
```
**Expected:** Free-form Q&A about the veteran's case

---

## Test 2: API Server

### Start the API Server
```bash
python vris_api_example.py
```

### What to Expect:
```
VRIS API Server Starting
System vectorstore pre-loaded and ready
Veteran documents will be processed on-demand
Files auto-delete after 72 hours

INFO: Uvicorn running on http://0.0.0.0:8000
```

### Test with Browser
Open: `http://localhost:8000/docs`

You'll see interactive Swagger UI to test all endpoints.

### Test with cURL

#### 1. Health Check
```bash
curl http://localhost:8000/api/vris/health
```
**Expected:**
```json
{
  "status": "healthy",
  "system_vectorstore_ready": true,
  "timestamp": "2026-02-01T..."
}
```

#### 2. Upload Documents
```bash
curl -X POST http://localhost:8000/api/vris/upload-documents \
  -F "files=@user_pdfs/1/sample_va_decision_letter.txt" \
  -F "files=@user_pdfs/1/sample_cp_exam_back.txt" \
  -F "files=@user_pdfs/1/sample_cp_exam_ptsd.txt" \
  -F "files=@user_pdfs/1/sample_private_sleep_study.txt"
```
**Expected:**
```json
{
  "success": true,
  "session_id": "20260201_143022_123456",
  "files_uploaded": 4,
  "files": [...],
  "expiry_hours": 72
}
```
**Save the `session_id` for next steps!**

#### 3. Free Snapshot Analysis
```bash
curl -X POST http://localhost:8000/api/vris/analyze/free-snapshot \
  -H "Content-Type: application/json" \
  -d '{"session_id": "YOUR_SESSION_ID_HERE"}'
```
**Expected:** High-level analysis JSON

#### 4. Second Look VRE
```bash
curl -X POST http://localhost:8000/api/vris/analyze/second-look \
  -H "Content-Type: application/json" \
  -d '{"session_id": "YOUR_SESSION_ID_HERE"}'
```
**Expected:** Full analysis with:
- VRIS-A extraction data
- VRIS-B reasoning with CFR citations
- Phase 1 and Phase 2 opportunities

#### 5. Custom Query
```bash
curl -X POST http://localhost:8000/api/vris/query/custom \
  -H "Content-Type: application/json" \
  -d '{
    "session_id": "YOUR_SESSION_ID_HERE",
    "query": "What evidence supports a 50% sleep apnea rating?",
    "pipeline": "vris-b"
  }'
```

#### 6. Delete Session
```bash
curl -X DELETE http://localhost:8000/api/vris/session/YOUR_SESSION_ID_HERE
```

---

## Test 3: Python Script Integration

Create `test_vris.py`:

```python
from vris_rag_system import VRISRAGSystem

# Initialize VRIS
vris = VRISRAGSystem(
    system_docs_folder="./system-doc",
    veteran_docs_folder="./user_pdfs/1",
    persist_directory="./chroma_db",
    model_name="gpt-4"
)

# Initialize (system docs load from disk, veteran docs process fresh)
vris.initialize()

# Test Second Look VRE
result = vris.generate_second_look_vre()

print("VRIS-A Extraction:")
print(result['vris_a_result'][:500])  # First 500 chars

print("\n\nVRIS-B Reasoning:")
print(result['vris_b_result'][:500])  # First 500 chars
```

Run:
```bash
python test_vris.py
```

---

## Test 4: Frontend Integration Test

### HTML Test Page

Create `test_frontend.html`:

```html
<!DOCTYPE html>
<html>
<head>
    <title>VRIS Test</title>
</head>
<body>
    <h1>VRIS Upload Test</h1>
    <input type="file" id="fileInput" multiple accept=".pdf,.docx,.txt">
    <button onclick="uploadFiles()">Upload & Analyze</button>
    <div id="results"></div>

    <script>
        async function uploadFiles() {
            const files = document.getElementById('fileInput').files;
            const formData = new FormData();
            
            for (let file of files) {
                formData.append('files', file);
            }
            
            // Upload
            const uploadResponse = await fetch('http://localhost:8000/api/vris/upload-documents', {
                method: 'POST',
                body: formData
            });
            const uploadData = await uploadResponse.json();
            console.log('Upload:', uploadData);
            
            // Analyze
            const analyzeResponse = await fetch('http://localhost:8000/api/vris/analyze/second-look', {
                method: 'POST',
                headers: {'Content-Type': 'application/json'},
                body: JSON.stringify({session_id: uploadData.session_id})
            });
            const analyzeData = await analyzeResponse.json();
            
            document.getElementById('results').innerHTML = 
                '<pre>' + JSON.stringify(analyzeData, null, 2) + '</pre>';
        }
    </script>
</body>
</html>
```

Open `test_frontend.html` in browser and test file upload.

---

## Expected Test Results

### ✅ What Should Happen:

1. **System Startup:**
   - System vectorstore loads instantly (42 chunks)
   - Veteran documents process fresh (~20 chunks)
   - Both pipelines initialize successfully

2. **Second Look VRE Analysis:**
   - Identifies PTSD underrating (30% → 50%)
   - Identifies missed Sleep Apnea (0% → 50%)
   - Identifies secondary Depression
   - Calculates potential combined rating increase

3. **Document Classification:**
   - Correctly identifies VA Decision Letter
   - Correctly identifies C&P Exams
   - Correctly identifies Private Medical Records

4. **API Responses:**
   - All endpoints return JSON successfully
   - Session management works
   - Files upload correctly

### ❌ Common Issues:

**Error: "OPENAI_API_KEY not found"**
- Solution: Create `.env` file with `OPENAI_API_KEY=sk-...`

**Error: "No documents found"**
- Solution: Ensure files are in correct folders
- System docs: `./system-doc/`
- Veteran docs: `./user_pdfs/1/`

**Error: "VRIS-A not initialized"**
- Solution: Ensure veteran documents loaded successfully

**API Error: "Session not found"**
- Solution: Check session_id is correct and hasn't expired

---

## Performance Benchmarks

**Expected Times:**
- System startup (first time): 30-60 seconds
- System startup (subsequent): 2-5 seconds
- Veteran doc processing: 10-20 seconds
- Analysis generation: 20-40 seconds per query
- API response: 25-50 seconds total

---

## Quick Test Command

```bash
# One-line test
python -c "from vris_rag_system import VRISRAGSystem; v = VRISRAGSystem('./system-doc', './user_pdfs/1', './chroma_db'); v.initialize(); print('✓ VRIS working!')"
```

---

## Troubleshooting

Run diagnostics:
```bash
python -c "
import os
print('System docs:', len(list(os.listdir('./system-doc'))))
print('Veteran docs:', len(list(os.listdir('./user_pdfs/1'))))
print('ChromaDB exists:', os.path.exists('./chroma_db/system'))
print('OpenAI key set:', bool(os.getenv('OPENAI_API_KEY')))
"
```

All checks should pass for successful testing!
