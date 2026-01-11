# PDF RAG FastAPI

A FastAPI-based REST API for the PDF RAG (Retrieval-Augmented Generation) system.

## Features

- 📄 **Base PDF Knowledge Base** - Query against PDFs stored in your project
- 👤 **User-Specific PDFs** - Upload and query user-specific documents
- 🔄 **Combined Queries** - Query both base and user PDFs together
- 🚀 **RESTful API** - Easy integration with any project
- 🔒 **User Isolation** - Each user's data is kept separate

## Quick Start

### 1. Install Dependencies

```bash
pip install -r api_requirements.txt
```

### 2. Setup Environment

Ensure your `.env` file has your OpenAI API key:
```
OPENAI_API_KEY=sk-your-api-key-here
```

### 3. Start the API Server

```bash
python api.py
```

The API will be available at `http://localhost:8000`

### 4. View API Documentation

Open your browser and go to:
- **Swagger UI**: http://localhost:8000/docs
- **ReDoc**: http://localhost:8000/redoc

## API Endpoints

### Health Check
```bash
GET /health
```

### Upload User PDF
```bash
POST /upload-user-pdf
Content-Type: multipart/form-data

Form fields:
- file: PDF file
- user_id: User identifier (required)
- user_name: User name (optional)
- user_email: User email (optional)
```

### Query RAG System
```bash
POST /query
Content-Type: application/json

{
  "question": "Your question here",
  "user_id": "user123",
  "use_user_pdf": true
}
```

### Query Base PDFs Only
```bash
POST /query-base-only
Content-Type: application/x-www-form-urlencoded

question=Your question here
```

### List Base PDFs
```bash
GET /list-base-pdfs
```

### List User PDFs
```bash
GET /list-user-pdfs/{user_id}
```

### Delete User Data
```bash
DELETE /delete-user-data/{user_id}
```

## Usage from Another Project

### Option 1: Using the Python Client

```python
from api_client import PDFRAGClient

# Initialize client
client = PDFRAGClient("http://localhost:8000")

# Upload user PDF
client.upload_user_pdf(
    pdf_path="document.pdf",
    user_id="user123",
    user_name="John Doe"
)

# Query with user PDF
result = client.query(
    question="What is mentioned in the document?",
    user_id="user123",
    use_user_pdf=True
)

print(result['answer'])
```

### Option 2: Using Requests Directly

```python
import requests

# Upload PDF
with open('document.pdf', 'rb') as f:
    files = {'file': f}
    data = {'user_id': 'user123'}
    response = requests.post(
        'http://localhost:8000/upload-user-pdf',
        files=files,
        data=data
    )

# Query
payload = {
    "question": "Your question?",
    "user_id": "user123",
    "use_user_pdf": True
}
response = requests.post(
    'http://localhost:8000/query',
    json=payload
)
result = response.json()
```

### Option 3: Using cURL

```bash
# Upload PDF
curl -X POST "http://localhost:8000/upload-user-pdf" \
  -F "file=@document.pdf" \
  -F "user_id=user123" \
  -F "user_name=John Doe"

# Query
curl -X POST "http://localhost:8000/query" \
  -H "Content-Type: application/json" \
  -d '{
    "question": "Your question?",
    "user_id": "user123",
    "use_user_pdf": true
  }'
```

## Architecture

```
┌─────────────────────┐
│  External Project   │
│  (Your Other App)   │
└──────────┬──────────┘
           │
           │ HTTP Requests
           │
           ▼
┌─────────────────────┐
│   FastAPI Server    │
│      (api.py)       │
└──────────┬──────────┘
           │
           ├──────────────────┐
           │                  │
           ▼                  ▼
┌──────────────────┐  ┌──────────────┐
│   Base PDFs      │  │  User PDFs   │
│  (pdfs folder)   │  │ (user_pdfs/) │
└──────────────────┘  └──────────────┘
           │                  │
           ▼                  ▼
┌──────────────────────────────────┐
│         ChromaDB Storage         │
│  (chroma_db_base, chroma_db_user)│
└──────────────────────────────────┘
```

## Workflow

1. **Base Knowledge**: PDFs in `pdfs/` folder are loaded on API startup
2. **User Upload**: External project uploads user-specific PDF via API
3. **Combined Processing**: System combines base PDFs + user PDF
4. **Query Processing**: Questions are answered using both sources
5. **Response**: Answer with source citations returned to external project

## Folder Structure

```
langchain-chromadb-gptapi-rag/
├── api.py                  # FastAPI application
├── api_client.py          # Python client for testing
├── pdf_rag_system.py      # Core RAG system
├── pdfs/                  # Base RAG PDFs (your knowledge base)
├── user_pdfs/             # User-uploaded PDFs (organized by user_id)
├── chroma_db_base/        # Vectorstore for base PDFs
└── chroma_db_user_*/      # Vectorstores for user-specific data
```

## Configuration

Edit these variables in `api.py`:

```python
BASE_PDF_FOLDER = "./pdfs"           # Your base PDFs
USER_PDF_FOLDER = "./user_pdfs"      # User uploads
CHROMA_DB_BASE = "./chroma_db_base"  # Base vectorstore
CHROMA_DB_USER = "./chroma_db_user"  # User vectorstores
```

## Testing

Run the test client:

```bash
python api_client.py
```

## Production Deployment

### Using Uvicorn

```bash
uvicorn api:app --host 0.0.0.0 --port 8000 --workers 4
```

### Using Gunicorn

```bash
gunicorn api:app -w 4 -k uvicorn.workers.UvicornWorker --bind 0.0.0.0:8000
```

### Docker (Optional)

Create a `Dockerfile`:
```dockerfile
FROM python:3.11-slim

WORKDIR /app

COPY api_requirements.txt .
RUN pip install -r api_requirements.txt

COPY . .

CMD ["uvicorn", "api:app", "--host", "0.0.0.0", "--port", "8000"]
```

Build and run:
```bash
docker build -t pdf-rag-api .
docker run -p 8000:8000 -v $(pwd)/pdfs:/app/pdfs pdf-rag-api
```

## Security Considerations

- Add authentication (JWT tokens, API keys)
- Implement rate limiting
- Validate file uploads (size, type)
- Sanitize user inputs
- Use HTTPS in production
- Set CORS policies appropriately

## Troubleshooting

### API not starting
- Check if port 8000 is available
- Verify OpenAI API key in `.env`
- Ensure base PDFs exist in `pdfs/` folder

### No answers returned
- Check if vectorstore is initialized
- Verify OpenAI API quota
- Ensure PDFs are properly loaded

### User data not found
- Verify user_id is correct
- Check if PDF was successfully uploaded
- Look for error messages in console

## License

MIT
