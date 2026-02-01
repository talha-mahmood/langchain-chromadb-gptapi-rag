# VRIS™ Requirements for Production

## Required Python Packages
```
langchain>=0.1.0
langchain-community>=0.0.13
langchain-openai>=0.0.2
langchain-chroma>=0.1.0
chromadb>=0.4.22
openai>=1.6.0
python-dotenv>=1.0.0
pypdf>=3.17.0
```

## Additional Production Dependencies

### For Azure OpenAI (if using Azure instead of OpenAI)
```
langchain-azure-openai
azure-identity
```

### For Enhanced Document Processing
```
pytesseract  # OCR for scanned documents
pdf2image  # Convert PDF to images for OCR
pillow  # Image processing
```

### For Agreement Scoring & NLP
```
sentence-transformers  # Semantic similarity
scikit-learn  # ML utilities
numpy  # Numerical operations
```

### For Data Validation & Security
```
pydantic>=2.0.0  # Data validation
cryptography  # Encryption for PII
```

### For Production Deployment
```
fastapi  # REST API
uvicorn  # ASGI server
pydantic-settings  # Configuration management
python-multipart  # File upload handling
celery  # Background task processing
redis  # Task queue and caching
```

## Installation Command
```bash
pip install -r vris_requirements.txt
```

## Environment Variables Required

Create a `.env` file with:
```
# OpenAI Configuration
OPENAI_API_KEY=sk-...

# Or for Azure OpenAI
AZURE_OPENAI_API_KEY=...
AZURE_OPENAI_ENDPOINT=https://your-endpoint.openai.azure.com/
AZURE_OPENAI_DEPLOYMENT_NAME=gpt-4
AZURE_OPENAI_API_VERSION=2024-02-15-preview

# Database Configuration
CHROMA_PERSIST_DIRECTORY=./chroma_db
SYSTEM_DOCS_FOLDER=./system-doc
VETERAN_DOCS_FOLDER=./user_pdfs

# Security (for production)
ENCRYPTION_KEY=...
FILE_RETENTION_HOURS=72

# Optional: Advanced Features
ENABLE_OCR=true
AGREEMENT_THRESHOLD=90
```

## System Requirements

- Python 3.9 or higher
- 8GB+ RAM (16GB recommended for large document processing)
- 10GB+ disk space (for ChromaDB and document storage)
- GPU optional (improves performance for OCR and embeddings)

## Notes for Abdul's Team

1. **Model Selection**: Code uses `gpt-4` by default. For production:
   - Use `gpt-4-turbo` for faster processing
   - Use `gpt-4-turbo-preview` for latest features
   - Consider Azure OpenAI for enterprise compliance

2. **ChromaDB**: Current implementation uses local persistence
   - For production: Consider ChromaDB Cloud or self-hosted server
   - Enable authentication and encryption

3. **Document Classification**: Current implementation is keyword-based
   - For production: Implement ML-based classification using fine-tuned model
   - Consider using GPT-4V for document type detection

4. **Agreement Scoring**: Placeholder implementation
   - Implement semantic similarity between VRIS-A and VRIS-B outputs
   - Use sentence transformers or OpenAI embeddings
   - Calculate overlap percentage for confidence scoring

5. **Security Considerations**:
   - Encrypt documents at rest and in transit
   - Implement 72-hour auto-deletion (use scheduled tasks)
   - Sanitize all outputs to prevent PII leakage
   - Audit logging for compliance

6. **Performance Optimization**:
   - Cache embeddings for system documents
   - Batch process veteran documents
   - Use async processing for API calls
   - Implement rate limiting for OpenAI API

7. **CFR Database**: 
   - System currently relies on RAG over PDF documents
   - Consider structured database for CFR sections and Diagnostic Codes
   - Implement versioning for CFR updates

8. **Testing**:
   - Test with anonymized veteran documents
   - Validate CFR references against official sources
   - Test agreement scoring with known cases
   - Load testing for concurrent users
