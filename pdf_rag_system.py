import os
from pathlib import Path
from typing import List
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_chroma import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

# Load environment variables
load_dotenv()

class PDFRAGSystem:
    """PDF RAG System using LangChain, ChromaDB, and OpenAI"""
    
    def __init__(self, pdf_folder: str, persist_directory: str = "./chroma_db"):
        """
        Initialize the RAG system
        
        Args:
            pdf_folder: Path to folder containing PDF files
            persist_directory: Directory to persist ChromaDB data
        """
        self.pdf_folder = pdf_folder
        self.persist_directory = persist_directory
        self.api_key = os.getenv("OPENAI_API_KEY")
        
        if not self.api_key:
            raise ValueError("OPENAI_API_KEY not found in environment variables")
        
        # Initialize embeddings and LLM
        self.embeddings = OpenAIEmbeddings(openai_api_key=self.api_key)
        self.llm = ChatOpenAI(
            model_name="gpt-3.5-turbo",
            temperature=0,
            openai_api_key=self.api_key
        )
        
        self.vectorstore = None
        self.qa_chain = None
        self.retriever = None
    
    def load_pdfs(self) -> List:
        """Load all PDF files from the specified folder"""
        pdf_files = list(Path(self.pdf_folder).glob("*.pdf"))
        
        if not pdf_files:
            raise ValueError(f"No PDF files found in {self.pdf_folder}")
        
        print(f"Found {len(pdf_files)} PDF file(s)")
        
        documents = []
        for pdf_file in pdf_files:
            print(f"Loading: {pdf_file.name}")
            loader = PyPDFLoader(str(pdf_file))
            documents.extend(loader.load())
        
        print(f"Loaded {len(documents)} pages in total")
        return documents
    
    def split_documents(self, documents: List) -> List:
        """Split documents into chunks"""
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            length_function=len,
        )
        
        chunks = text_splitter.split_documents(documents)
        print(f"Split into {len(chunks)} chunks")
        return chunks
    
    def create_vectorstore(self, chunks: List):
        """Create or load ChromaDB vectorstore"""
        print("Creating vector embeddings and storing in ChromaDB...")
        
        self.vectorstore = Chroma.from_documents(
            documents=chunks,
            embedding=self.embeddings,
            persist_directory=self.persist_directory
        )
        
        print(f"Vectorstore created with {self.vectorstore._collection.count()} documents")
    
    def load_existing_vectorstore(self):
        """Load existing vectorstore from disk"""
        if not os.path.exists(self.persist_directory):
            raise ValueError(f"No existing vectorstore found at {self.persist_directory}")
        
        print("Loading existing vectorstore...")
        self.vectorstore = Chroma(
            persist_directory=self.persist_directory,
            embedding_function=self.embeddings
        )
        print(f"Loaded vectorstore with {self.vectorstore._collection.count()} documents")
    
    def setup_qa_chain(self):
        """Setup the QA chain with custom prompt using LCEL"""
        # Setup retriever
        self.retriever = self.vectorstore.as_retriever(
            search_type="similarity",
            search_kwargs={"k": 4}
        )
        
        # Custom prompt template
        template = """You are an assistant for question-answering tasks. 
Use the following pieces of retrieved context to answer the question. 
If you don't know the answer, just say that you don't know. 
Keep the answer concise.

Context:
{context}

Question: {question}

Answer:"""
        
        prompt = ChatPromptTemplate.from_template(template)
        
        # Helper function to format documents
        def format_docs(docs):
            return "\n\n".join(doc.page_content for doc in docs)
        
        # Create the RAG chain using LCEL
        self.qa_chain = (
            {
                "context": self.retriever | format_docs,
                "question": RunnablePassthrough()
            }
            | prompt
            | self.llm
            | StrOutputParser()
        )
        
        print("QA chain setup complete")
    
    def query(self, question: str) -> dict:
        """
        Query the RAG system
        
        Args:
            question: The question to ask
            
        Returns:
            Dictionary with answer and source documents
        """
        if not self.qa_chain:
            raise ValueError("QA chain not initialized. Run setup_qa_chain() first")
        
        print(f"\nQuestion: {question}")
        
        # Get the answer
        answer = self.qa_chain.invoke(question)
        
        # Get source documents separately
        source_docs = self.retriever.invoke(question)
        
        print(f"\nAnswer: {answer}")
        print(f"\nSource Documents ({len(source_docs)}):")
        for i, doc in enumerate(source_docs, 1):
            source = doc.metadata.get('source', 'Unknown')
            page = doc.metadata.get('page', 'Unknown')
            print(f"  {i}. Source: {Path(source).name}, Page: {page}")
        
        return {
            "answer": answer,
            "source_documents": source_docs
        }
    
    def initialize(self, force_reload: bool = False):
        """
        Initialize the RAG system
        
        Args:
            force_reload: If True, reload PDFs even if vectorstore exists
        """
        if force_reload or not os.path.exists(self.persist_directory):
            # Load and process PDFs
            documents = self.load_pdfs()
            chunks = self.split_documents(documents)
            self.create_vectorstore(chunks)
        else:
            # Load existing vectorstore
            self.load_existing_vectorstore()
        
        # Setup QA chain
        self.setup_qa_chain()
        print("\n" + "="*50)
        print("RAG System Ready!")
        print("="*50 + "\n")


def main():
    """Main function to demonstrate the RAG system"""
    
    # Configuration
    PDF_FOLDER = "./pdfs"  # Folder containing your PDF files
    CHROMA_DB_DIR = "./chroma_db"  # Directory to store ChromaDB
    
    # Create PDF folder if it doesn't exist
    os.makedirs(PDF_FOLDER, exist_ok=True)
    
    # Initialize RAG system
    rag = PDFRAGSystem(pdf_folder=PDF_FOLDER, persist_directory=CHROMA_DB_DIR)
    
    # Initialize (set force_reload=True to reprocess PDFs)
    rag.initialize(force_reload=False)
    
    # Example queries
    print("You can now ask questions about your PDFs!\n")
    
    # Interactive mode
    while True:
        question = input("\nEnter your question (or 'quit' to exit): ").strip()
        
        if question.lower() in ['quit', 'exit', 'q']:
            print("Goodbye!")
            break
        
        if not question:
            continue
        
        try:
            rag.query(question)
        except Exception as e:
            print(f"Error: {e}")


if __name__ == "__main__":
    main()
