# PDF RAG System with LangChain, ChromaDB, and OpenAI

A Retrieval-Augmented Generation (RAG) system that allows you to query multiple PDF documents using LangChain, ChromaDB for vector storage, and OpenAI's GPT models.

## Features

- 📄 Load and process multiple PDF files
- 🔍 Semantic search across all documents
- 💬 Natural language Q&A with source citations
- 💾 Persistent vector storage with ChromaDB
- 🔄 Efficient document chunking and embedding
- 🎯 Source document tracking with page numbers

## Prerequisites

- Python 3.8 or higher
- OpenAI API key

## Installation

1. **Clone or navigate to the project directory**

2. **Create a virtual environment (recommended)**
```bash
python -m venv venv

# On Windows
venv\Scripts\activate

# On macOS/Linux
source venv/bin/activate
```

3. **Install dependencies**
```bash
pip install -r requirements.txt
```

4. **Set up environment variables**
   - Copy `.env.example` to `.env`
   - Add your OpenAI API key:
   ```
   OPENAI_API_KEY=sk-your-actual-api-key-here
   ```

## Usage

### 1. Prepare Your PDFs

Create a `pdfs` folder in the project directory and add your PDF files:

```
langchain-chromadb-gptapi-rag/
├── pdfs/
│   ├── document1.pdf
│   ├── document2.pdf
│   └── document3.pdf
├── pdf_rag_system.py
├── requirements.txt
└── .env
```

### 2. Run the System

```bash
python pdf_rag_system.py
```

The system will:
- Load all PDF files from the `pdfs` folder
- Split documents into chunks
- Create embeddings using OpenAI
- Store vectors in ChromaDB
- Start an interactive Q&A session

### 3. Ask Questions

Once initialized, you can ask questions about your documents:

```
Enter your question (or 'quit' to exit): What is the main topic discussed in the documents?
```

The system will:
- Search for relevant document chunks
- Generate an answer using GPT-3.5-turbo
- Show source documents with page numbers

## Code Structure

### Main Components

```python
# Initialize the RAG system
rag = PDFRAGSystem(
    pdf_folder="./pdfs",
    persist_directory="./chroma_db"
)

# Load and process PDFs (first time only)
rag.initialize(force_reload=False)

# Query the system
result = rag.query("Your question here")
```

### Key Methods

- **`load_pdfs()`** - Load all PDFs from the specified folder
- **`split_documents()`** - Split documents into manageable chunks
- **`create_vectorstore()`** - Create embeddings and store in ChromaDB
- **`setup_qa_chain()`** - Initialize the QA chain with retrieval
- **`query()`** - Ask questions and get answers with sources

## Configuration

### Customize Chunk Size

Edit in [pdf_rag_system.py](pdf_rag_system.py):
```python
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,      # Adjust chunk size
    chunk_overlap=200,    # Adjust overlap
    length_function=len,
)
```

### Change OpenAI Model

Edit in [pdf_rag_system.py](pdf_rag_system.py):
```python
self.llm = ChatOpenAI(
    model_name="gpt-4",  # Use GPT-4 instead of GPT-3.5-turbo
    temperature=0,
    openai_api_key=self.api_key
)
```

### Adjust Retrieval Parameters

Edit in [pdf_rag_system.py](pdf_rag_system.py):
```python
retriever=self.vectorstore.as_retriever(
    search_type="similarity",
    search_kwargs={"k": 4}  # Number of documents to retrieve
)
```

## How It Works

1. **Document Loading**: PDFs are loaded using PyPDFLoader
2. **Text Splitting**: Documents are split into chunks with overlap
3. **Embedding**: Each chunk is converted to vector embeddings using OpenAI
4. **Vector Storage**: Embeddings are stored in ChromaDB for fast retrieval
5. **Query Processing**: User questions are embedded and matched against stored vectors
6. **Answer Generation**: Relevant chunks are sent to GPT along with the question
7. **Source Citation**: Original documents and page numbers are tracked and returned

## Persistence

The vectorstore is saved to disk in the `chroma_db` directory. On subsequent runs:
- If `force_reload=False`: Uses existing vectorstore (fast)
- If `force_reload=True`: Reprocesses all PDFs (use when adding new documents)

## Example Session

```
Found 3 PDF file(s)
Loading: research_paper.pdf
Loading: user_manual.pdf
Loading: report.pdf
Loaded 45 pages in total
Split into 120 chunks
Creating vector embeddings and storing in ChromaDB...
Vectorstore created with 120 documents
QA chain setup complete

==================================================
RAG System Ready!
==================================================

You can now ask questions about your PDFs!

Enter your question (or 'quit' to exit): What are the key findings?

Question: What are the key findings?

Answer: Based on the documents, the key findings include...

Source Documents (4):
  1. Source: research_paper.pdf, Page: 12
  2. Source: report.pdf, Page: 5
  3. Source: research_paper.pdf, Page: 15
  4. Source: user_manual.pdf, Page: 8
```

## Troubleshooting

### No PDF files found
- Ensure your PDFs are in the `pdfs` folder
- Check that files have `.pdf` extension

### OpenAI API Key Error
- Verify your API key in the `.env` file
- Ensure you have credits in your OpenAI account

### Memory Issues
- Reduce `chunk_size` for large documents
- Process fewer PDFs at once

## Cost Considerations

- **Embedding**: OpenAI charges per token for embeddings
- **Queries**: Each question uses GPT-3.5-turbo tokens
- **Tip**: Use `force_reload=False` to avoid re-embedding on every run

## License

MIT License

## Contributing

Feel free to submit issues and enhancement requests!
