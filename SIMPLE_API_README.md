# VRIS™ Simple API - Single Endpoint

## Overview

This is the simplified, production-ready VRIS API with **ONE endpoint**:
- Upload veteran documents → Get complete analysis
- No sessions, no multiple steps, no Q&A
- Just: **Documents In → Analysis Out**

## How It Works

```
Your Frontend → POST /api/vris/analyze (with PDFs) → Complete VRIS Analysis Response
```

### What VRIS Does:

1. **VRIS-A (Extraction)**
   - Extracts diagnostic codes, CFR references, symptoms
   - Identifies current ratings and conditions
   - Pulls all medical evidence

2. **VRIS-B (Reasoning)**
   - Cross-checks against VA rating rules (Title 38 CFR)
   - Identifies underrated conditions
   - Finds missed conditions
   - Flags secondary condition opportunities

3. **Reconciliation**
   - Compares VRIS-A and VRIS-B findings
   - High confidence when 90%+ agreement
   - Structured response with all findings

## Installation

```bash
# Install dependencies
pip install -r vris_requirements.txt

# Ensure .env file has OpenAI API key
echo "OPENAI_API_KEY=sk-your-key-here" > .env
```

## Running the API

### Start the Server
```bash
python vris_simple_api.py
```

Server starts at: `http://localhost:8000`

### What Happens at Startup:
```
🚀 VRIS™ API Server Starting...
Loading VRIS system documentation and CFR rules...
✓ System vectorstore ready
✓ VRIS ready to analyze veteran documents
```

The system documentation (CFR rules, VRIS docs) loads **once** at startup and is cached. Veteran documents are processed fresh for each request (not stored).

## API Usage

### Single Endpoint: `/api/vris/analyze`

**Method:** `POST`
**Content-Type:** `multipart/form-data`
**Input:** Upload veteran PDF/DOCX/TXT files
**Output:** Complete VRIS analysis JSON

### Request Example (cURL)

```bash
curl -X POST http://localhost:8000/api/vris/analyze \
  -F "files=@va_decision_letter.pdf" \
  -F "files=@cp_exam.pdf" \
  -F "files=@medical_records.pdf"
```

### Request Example (Python)

```python
import requests

files = [
    ('files', ('decision_letter.pdf', open('decision_letter.pdf', 'rb'), 'application/pdf')),
    ('files', ('cp_exam.pdf', open('cp_exam.pdf', 'rb'), 'application/pdf'))
]

response = requests.post('http://localhost:8000/api/vris/analyze', files=files)
result = response.json()

print(f"Total Opportunities: {result['summary']['total_opportunities_identified']}")
print(f"Underrated Conditions: {result['summary']['underrated_conditions_count']}")
print(f"Missed Conditions: {result['summary']['missed_conditions_count']}")
```

### Request Example (JavaScript/React)

```javascript
const analyzeDocuments = async (files) => {
  const formData = new FormData();
  
  files.forEach(file => {
    formData.append('files', file);
  });
  
  const response = await fetch('http://localhost:8000/api/vris/analyze', {
    method: 'POST',
    body: formData
  });
  
  const result = await response.json();
  return result;
};

// Usage
const files = document.getElementById('fileInput').files;
const analysis = await analyzeDocuments(files);
console.log(analysis.summary);
```

## Response Format

```json
{
  "success": true,
  "timestamp": "2026-02-01T14:30:22.123456",
  "analysis_type": "VRIS Complete Analysis",
  
  "files_analyzed": [
    {"filename": "decision_letter.pdf", "size": 245678, "type": "pdf"},
    {"filename": "cp_exam.pdf", "size": 189234, "type": "pdf"}
  ],
  
  "vris_a_extraction": {
    "description": "Extracted data from veteran documents",
    "data": "1. CURRENT VA RATING DATA:\n   - Combined disability rating: 60%\n   ..."
  },
  
  "vris_b_reasoning": {
    "description": "Analysis against VA rating criteria (38 CFR)",
    "data": "1. PTSD (Current Rating: 30%)\n   - Evidence: Clinical presentation suggests 50%...\n   ..."
  },
  
  "findings": {
    "underrated_conditions": [
      {
        "name": "PTSD",
        "current_rating": "30%",
        "potential_rating": "50%",
        "confidence": 90,
        "phase": 1,
        "evidence": ["Multiple sick days from work...", "Declining work performance..."],
        "cfr_citations": ["38 CFR 4.130"]
      }
    ],
    "missed_conditions": [
      {
        "name": "Sleep Apnea Secondary to PTSD",
        "current_rating": "Not Rated",
        "potential_rating": "50%",
        "confidence": 85,
        "phase": 1,
        "evidence": ["Requires CPAP nightly...", "Sleep study shows AHI 22..."],
        "cfr_citations": ["DC 6847", "38 CFR 3.310(a)"]
      }
    ],
    "secondary_conditions": [
      {
        "name": "Depression Secondary to PTSD",
        "confidence": 80,
        "phase": 2
      }
    ],
    "total_opportunities": 3
  },
  
  "summary": {
    "total_opportunities_identified": 3,
    "underrated_conditions_count": 1,
    "missed_conditions_count": 1,
    "secondary_conditions_count": 1,
    "high_confidence_opportunities": 2,
    "recommendation": "Strong increase opportunities identified"
  },
  
  "data_retention": "Files processed in-memory only. Not stored. Deleted immediately after analysis.",
  "system_info": "VRIS™ - Veteran Rating Intelligence System"
}
```

## Testing

### Option 1: Test Script (Automated)

```bash
python test_simple_api.py
```

This will:
1. Check API health
2. Upload sample documents
3. Display analysis results

### Option 2: Interactive API Docs

1. Start the API: `python vris_simple_api.py`
2. Open browser: `http://localhost:8000/docs`
3. Click "Try it out" on `/api/vris/analyze`
4. Upload files and execute

### Option 3: Manual cURL Test

```bash
# Health check
curl http://localhost:8000/api/vris/health

# Analyze documents
curl -X POST http://localhost:8000/api/vris/analyze \
  -F "files=@user_pdfs/1/sample_va_decision_letter.txt" \
  -F "files=@user_pdfs/1/sample_cp_exam_ptsd.txt"
```

## Expected Results with Sample Documents

Using the provided sample documents, VRIS should identify:

### Underrated Conditions:
- **PTSD**: 30% → 50% (90% confidence)
  - Evidence: Work impairment, relationship strain, sleep issues
  - CFR: 38 CFR 4.130 criteria for 50% clearly met

### Missed Conditions:
- **Sleep Apnea**: Not rated → 50% (85% confidence)
  - Evidence: Moderate OSA requiring CPAP
  - Secondary to: PTSD and chronic pain
  - CFR: DC 6847

### Secondary Conditions:
- **Depression**: Secondary to PTSD (80% confidence)
- **Radiculopathy**: Related to back condition

### Estimated Impact:
- Current: 60% combined
- Potential: 80-90% combined
- Monthly increase: ~$1,000-1,500
- Annual increase: ~$12,000-18,000

## Performance

- **System startup**: 5-10 seconds (loads CFR rules)
- **Per-analysis time**: 30-60 seconds
- **Concurrent requests**: Supports multiple simultaneous analyses
- **File size limits**: Default 100MB per file (configurable)

## Data Privacy & Security

✅ **In-Memory Processing**: Documents processed in RAM only
✅ **Immediate Deletion**: Files deleted right after analysis
✅ **No Storage**: Nothing persisted to disk
✅ **Encryption**: Files encrypted during transfer (HTTPS in production)
✅ **Compliance**: HIPAA-ready with Azure OpenAI BAA

## Production Deployment

### Environment Variables
```bash
OPENAI_API_KEY=sk-your-key
# or for Azure OpenAI
AZURE_OPENAI_API_KEY=your-key
AZURE_OPENAI_ENDPOINT=https://your-endpoint.openai.azure.com/
AZURE_OPENAI_DEPLOYMENT_NAME=gpt-4
```

### Docker Deployment
```dockerfile
FROM python:3.11-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install -r vris_requirements.txt
COPY . .
CMD ["python", "vris_simple_api.py"]
```

### Production Settings
```python
# In vris_simple_api.py, update:
uvicorn.run(
    app, 
    host="0.0.0.0", 
    port=8000,
    workers=4,  # Multiple workers for concurrency
    log_level="info",
    ssl_keyfile="/path/to/key.pem",  # HTTPS
    ssl_certfile="/path/to/cert.pem"
)
```

## Troubleshooting

### Error: "Connection refused"
- **Solution**: Ensure API is running: `python vris_simple_api.py`

### Error: "OPENAI_API_KEY not found"
- **Solution**: Create `.env` file with API key

### Error: "System vectorstore not ready"
- **Solution**: Ensure `system-doc/` folder has VRIS documentation

### Slow response times
- **Solution**: Use `gpt-4-turbo` instead of `gpt-4` for faster processing
- **Solution**: Increase workers: `workers=4` in uvicorn.run()

## Support

For issues or questions:
- Check logs in terminal where API is running
- Verify files are valid PDFs/DOCX/TXT
- Ensure OpenAI API has sufficient credits
- Check `http://localhost:8000/docs` for API documentation

## License

Proprietary - VRIS™ / Get VA Help Platform
