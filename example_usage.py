"""
Example usage of the PDF RAG system.
This script demonstrates how to use the PDFRAGSystem class programmatically.
"""

from pdf_rag_system import PDFRAGSystem

def main():
    # Configuration
    PDF_FOLDER = "./pdfs"
    CHROMA_DB_DIR = "./chroma_db"
    
    print("="*60)
    print("PDF RAG System - Example Usage")
    print("="*60)
    
    # Initialize the RAG system
    print("\n1. Initializing RAG System...")
    rag = PDFRAGSystem(
        pdf_folder=PDF_FOLDER,
        persist_directory=CHROMA_DB_DIR
    )
    
    # Load and process PDFs
    # Set force_reload=True to reprocess PDFs if you've added new documents
    # Set force_reload=False to use existing vectorstore (faster)
    print("\n2. Processing PDFs...")
    rag.initialize(force_reload=False)
    
    # Example queries
    questions = [
        "What are the main topics covered in these documents?",
        "Can you summarize the key findings?",
        "What recommendations are provided?"
    ]
    
    print("\n3. Asking questions...")
    print("="*60)
    
    for i, question in enumerate(questions, 1):
        print(f"\n[Question {i}]")
        result = rag.query(question)
        print("\n" + "-"*60)
        
        # You can access the result components:
        # result['answer'] - The answer
        # result['source_documents'] - The source documents used
    
    print("\n" + "="*60)
    print("Example completed!")
    print("="*60)


if __name__ == "__main__":
    main()
