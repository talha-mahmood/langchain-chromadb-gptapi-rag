# VRIS™ RAG System - Implementation Guide

## Overview

This implementation provides the RAG (Retrieval Augmented Generation) component for VRIS™ (Veteran Rating Intelligence System), a dual-pipeline AI system that analyzes veteran disability documentation to identify underratings, missed conditions, and secondary conditions.

## Architecture

### Dual-Pipeline System

**VRIS-A: Extraction Pipeline**
- Extracts structured data from veteran documents
- Identifies conditions, symptoms, diagnostic codes, CFR references
- Classifies documents by type (VA Decision Letter, C&P Exam, DBQs, etc.)
- Creates normalized representation of veteran's medical profile

**VRIS-B: Reasoning Pipeline**
- Analyzes extracted data against VA rating criteria (38 CFR)
- Identifies underratings and missed conditions
- Maps secondary condition relationships
- Calculates potential rating increases with confidence scores
- Provides CFR citations and evidence mappings

### Agreement Reconciliation

When both VRIS-A and VRIS-B agree by 90%+ on a finding, it's treated as a high-confidence opportunity.

## File Structure

```
langchain-chromadb-gptapi-rag/
├── vris_rag_system.py          # Main VRIS RAG implementation
├── vris_example_usage.py       # Example workflows and demonstrations
├── vris_requirements.txt       # Python dependencies
├── VRIS_PRODUCTION_REQUIREMENTS.md  # Production deployment notes
├── system-doc/                 # VRIS system documentation (CFR rules, specs)
│   ├── VRIS_Gap_Analysis_Pricing_Structure.pdf
│   ├── VRIS_Free_Analysis_Sample_Anonymized.pdf
│   ├── VRIS_Document_Requirements_and_Secondary_Questionnaire.pdf
│   └── VRIS_Backend_TechSpec_Abdul_v1.pdf
├── user_pdfs/                  # Veteran document uploads
│   └── 1/                      # Per-veteran folders
├── chroma_db/                  # Vector database storage
│   ├── system/                 # System documentation vectors
│   └── veteran/                # Veteran document vectors
└── .env                        # Environment configuration
```

## Installation

### 1. Install Dependencies

```bash
pip install -r vris_requirements.txt
```

### 2. Configure Environment

Create a `.env` file:

```env
OPENAI_API_KEY=sk-your-openai-api-key
```

For Azure OpenAI:
```env
AZURE_OPENAI_API_KEY=your-azure-key
AZURE_OPENAI_ENDPOINT=https://your-endpoint.openai.azure.com/
AZURE_OPENAI_DEPLOYMENT_NAME=gpt-4
AZURE_OPENAI_API_VERSION=2024-02-15-preview
```

### 3. Prepare Documents

**System Documents** (place in `system-doc/`):
- VRIS documentation
- CFR sections and rating criteria
- VA M21-1 Manual sections
- Diagnostic Code references

**Veteran Documents** (place in `user_pdfs/1/` or per-veteran folders):
- VA Rating Decision Letter
- C&P Examination reports
- Service Treatment Records (STRs)
- Private medical records
- DBQs, Nexus letters, etc.

## Usage

### Basic Usage

```python
from vris_rag_system import VRISRAGSystem

# Initialize VRIS
vris = VRISRAGSystem(
    system_docs_folder="./system-doc",
    veteran_docs_folder="./user_pdfs/1",
    persist_directory="./chroma_db",
    model_name="gpt-4"
)

# Initialize system (loads documents, creates embeddings)
vris.initialize(force_reload=False)

# Generate Free Rating Snapshot
snapshot = vris.generate_free_snapshot()
print(snapshot)

# Generate Initial VRE (for first-time filers)
initial_vre = vris.generate_initial_vre()
print(initial_vre)

# Generate Second Look VRE (for already-rated veterans)
second_look = vris.generate_second_look_vre()
print(second_look)
```

### Interactive Mode

```bash
python vris_rag_system.py
```

This launches an interactive menu with options for:
1. Free Rating Snapshot
2. Initial VRE (First-Time Filer)
3. Second Look VRE (Already-Rated)
4. Custom VRIS-A Extraction
5. Custom VRIS-B Analysis
6. Interactive Query Mode

### Running Examples

```bash
# Run all examples
python vris_example_usage.py

# Run specific example (1-7)
python vris_example_usage.py 1
```

## Key Features

### Document Classification

VRIS automatically classifies uploaded documents:
- VA Rating Decision Letter
- C&P Examination Reports
- Disability Benefits Questionnaires (DBQs)
- Service Treatment Records (STRs)
- Private Medical Records
- Buddy Statements
- Nexus Letters
- DD-214
- Code Sheets

### Evaluation Modes

**1. Free Rating Snapshot**
- High-level analysis without detailed CFR references
- Identifies potential underratings and missed conditions
- Estimates rating increase range
- Recommends next steps

**2. Initial VRE (First-Time Filer)**
- Identifies all service-connected conditions
- Suggests initial rating bands
- Maps secondary conditions
- Provides claim language frameworks

**3. Second Look VRE (Already-Rated Veteran)**
- Compares current rating to evidence
- Identifies underrated conditions
- Finds missed conditions and secondary opportunities
- Provides detailed CFR citations

### Custom Queries

**VRIS-A Extraction:**
```python
result = vris.vris_a_extract("""
Extract all information about knee conditions:
- Current rating and diagnostic code
- ROM measurements
- Pain levels and frequency
- Functional limitations
""")
```

**VRIS-B Reasoning:**
```python
result = vris.vris_b_analyze("""
Analyze if the knee condition evidence supports a higher rating.
Compare to 38 CFR 4.71a criteria.
Identify any secondary conditions (hip, back, gait).
Provide confidence score and CFR citations.
""")
```

### Dual Pipeline Analysis

```python
result = vris.dual_pipeline_analysis(
    extraction_query="Extract all rated conditions with severity evidence",
    reasoning_query="Analyze each condition for potential rating increases"
)

print(f"VRIS-A: {result['vris_a_extraction']}")
print(f"VRIS-B: {result['vris_b_reasoning']}")
print(f"Agreement: {result['agreement_score']}%")
```

## API Integration (For Production)

### FastAPI Example

```python
from fastapi import FastAPI, UploadFile, File
from vris_rag_system import VRISRAGSystem

app = FastAPI()

@app.post("/api/vris/analyze/snapshot")
async def free_snapshot(files: List[UploadFile] = File(...)):
    # Save uploaded files
    # Initialize VRIS
    vris = VRISRAGSystem(...)
    vris.initialize()
    
    # Generate snapshot
    result = vris.generate_free_snapshot()
    return result

@app.post("/api/vris/analyze/initial-vre")
async def initial_vre(files: List[UploadFile] = File(...)):
    vris = VRISRAGSystem(...)
    vris.initialize()
    result = vris.generate_initial_vre()
    return result

@app.post("/api/vris/analyze/second-look")
async def second_look_vre(files: List[UploadFile] = File(...)):
    vris = VRISRAGSystem(...)
    vris.initialize()
    result = vris.generate_second_look_vre()
    return result
```

## Performance Considerations

### Embeddings Generation
- System documents: Process once, cache indefinitely
- Veteran documents: Process per-session, delete after 72 hours

### Model Selection
- **gpt-4**: Best accuracy for medical/legal analysis (recommended)
- **gpt-4-turbo**: Faster processing, slightly lower accuracy
- **gpt-3.5-turbo**: Faster/cheaper but not recommended for VRIS

### Caching Strategy
- Cache system document embeddings
- Use ChromaDB persistence to avoid reprocessing
- Implement Redis for session caching

### Rate Limiting
- OpenAI API: 3,500 requests/minute (Tier 2)
- Batch veteran documents when possible
- Use exponential backoff for retries

## Security & Compliance

### Data Protection
1. **Encryption**: Encrypt documents at rest and in transit
2. **Auto-Deletion**: Purge veteran files after 72 hours
3. **PII Sanitization**: Remove PII from logs and outputs
4. **Vault Storage**: Optional long-term storage with explicit consent

### HIPAA Compliance
- Use Azure OpenAI with BAA for healthcare compliance
- Implement audit logging
- Access controls and authentication
- Data residency considerations

## Testing

### Unit Tests
```bash
pytest tests/test_vris_extraction.py
pytest tests/test_vris_reasoning.py
pytest tests/test_document_classifier.py
```

### Integration Tests
```bash
pytest tests/test_vris_full_pipeline.py
```

### Load Testing
```bash
locust -f tests/load_test.py
```

## Troubleshooting

### Common Issues

**Issue: "No documents found"**
- Check that PDFs exist in specified folders
- Verify folder paths are correct
- Ensure PDFs are not corrupted

**Issue: "OPENAI_API_KEY not found"**
- Create `.env` file with API key
- Verify `.env` is in project root
- Check key is valid and has credits

**Issue: "Vectorstore empty"**
- Run with `force_reload=True`
- Check PDF loading succeeded
- Verify ChromaDB permissions

**Issue: Low confidence scores**
- Ensure complete documentation provided
- Check document quality (OCR may be needed for scans)
- Verify system documents include CFR references

## Future Enhancements

### Phase 1 (MVP)
- [x] Dual-pipeline architecture
- [x] Document classification
- [x] Three evaluation modes
- [ ] Agreement scoring implementation
- [ ] Production API

### Phase 2
- [ ] ML-based document classification
- [ ] Automated CFR database updates
- [ ] Batch processing
- [ ] VSO dashboard integration

### Phase 3
- [ ] Multi-language support
- [ ] Voice input for questionnaires
- [ ] Automated DBQ generation
- [ ] VSO collaboration tools

## Support

For issues or questions about this implementation, contact:
- Development Team: Abdul and team
- Documentation: See VRIS_Backend_TechSpec_Abdul_v1.pdf

## License

Proprietary - Get VA Help / VRIS™ System
