# VRIS™ Simple API - Quick Start

## 🎯 One Endpoint, Complete Analysis

```
POST /api/vris/analyze
├── Input: Upload PDFs (VA docs, C&P exams, medical records)
└── Output: Complete VRIS analysis with all findings
```

## 🚀 Quick Start (3 Steps)

### Step 1: Start API Server
```bash
python vris_simple_api.py
```
✅ Server ready at `http://localhost:8000`

### Step 2: Upload Documents
```bash
# Option A: Use test script
python test_simple_api.py

# Option B: Use cURL
curl -X POST http://localhost:8000/api/vris/analyze \
  -F "files=@decision_letter.pdf" \
  -F "files=@cp_exam.pdf"

# Option C: Use browser
# Open http://localhost:8000/docs
# Click "Try it out" → Upload files → Execute
```

### Step 3: Get Results
```json
{
  "summary": {
    "total_opportunities_identified": 3,
    "underrated_conditions_count": 1,
    "missed_conditions_count": 1,
    "high_confidence_opportunities": 2
  },
  "findings": {
    "underrated_conditions": [...],
    "missed_conditions": [...],
    "secondary_conditions": [...]
  }
}
```

## 📋 What Your Senior Will Do

```javascript
// Frontend code
const uploadToVRIS = async (veteranFiles) => {
  const formData = new FormData();
  veteranFiles.forEach(file => formData.append('files', file));
  
  const response = await fetch('http://your-api/api/vris/analyze', {
    method: 'POST',
    body: formData
  });
  
  return await response.json();
};

// Usage
const analysis = await uploadToVRIS(selectedFiles);
console.log('Opportunities:', analysis.summary.total_opportunities_identified);
console.log('Underrated:', analysis.findings.underrated_conditions);
console.log('Missed:', analysis.findings.missed_conditions);
```

## ✅ What You Get

### VRIS-A Output (Extraction)
- All current VA ratings and diagnostic codes
- Symptoms and severity indicators
- Medical evidence citations
- Temporal relationships

### VRIS-B Output (Reasoning)
- Underrated conditions with CFR citations
- Missed conditions not currently rated
- Secondary condition opportunities
- Confidence scores (0-100%)
- Phase 1 (90%+ confidence) vs Phase 2

### Structured Findings
```json
{
  "name": "PTSD",
  "current_rating": "30%",
  "potential_rating": "50%",
  "confidence": 90,
  "phase": 1,
  "cfr_citations": ["38 CFR 4.130"]
}
```

## 🔒 Privacy & Security

✅ Documents processed in-memory only
✅ No storage to disk
✅ Deleted immediately after analysis
✅ 72-hour retention policy (in-memory only during processing)
✅ Ready for HIPAA compliance

## 📊 Sample Response Structure

```
{
  success: true
  timestamp: "2026-02-01T14:30:22"
  files_analyzed: [...]
  
  vris_a_extraction: {
    data: "Raw extracted information..."
  }
  
  vris_b_reasoning: {
    data: "CFR-backed analysis..."
  }
  
  findings: {
    underrated_conditions: [
      {name, current_rating, potential_rating, confidence, phase}
    ],
    missed_conditions: [
      {name, potential_rating, confidence, phase}
    ],
    secondary_conditions: [
      {name, confidence, phase}
    ]
  }
  
  summary: {
    total_opportunities_identified: 3,
    high_confidence_opportunities: 2,
    recommendation: "Strong increase opportunities identified"
  }
}
```

## 🧪 Test Now

```bash
# Terminal 1: Start API
python vris_simple_api.py

# Terminal 2: Test it
python test_simple_api.py
```

Expected output:
```
📊 SUMMARY:
   Total Opportunities: 3
   Underrated Conditions: 1
   Missed Conditions: 1
   High Confidence (90%+): 2

🔍 UNDERRATED CONDITIONS:
   • PTSD
     Current: 30%
     Potential: 50%
     Confidence: 90%
```

## 🌐 Production URLs

```bash
# Development
http://localhost:8000/api/vris/analyze

# Production (example)
https://api.getvahhelp.org/api/vris/analyze
```

That's it! One endpoint, complete analysis. 🎖️
