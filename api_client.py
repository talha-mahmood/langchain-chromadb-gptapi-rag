"""
Test client for the PDF RAG API.
Demonstrates how to interact with the API from another project.
"""

import requests
from pathlib import Path


class PDFRAGClient:
    """Client for interacting with the PDF RAG API"""
    
    def __init__(self, base_url: str = "http://localhost:8000"):
        """
        Initialize the client
        
        Args:
            base_url: Base URL of the API
        """
        self.base_url = base_url
    
    def health_check(self):
        """Check API health"""
        response = requests.get(f"{self.base_url}/health")
        return response.json()
    
    def upload_user_pdf(
        self,
        pdf_path: str,
        user_id: str,
        user_name: str = None,
        user_email: str = None
    ):
        """
        Upload a user-specific PDF
        
        Args:
            pdf_path: Path to the PDF file
            user_id: User identifier
            user_name: Optional user name
            user_email: Optional user email
        """
        with open(pdf_path, 'rb') as f:
            files = {'file': (Path(pdf_path).name, f, 'application/pdf')}
            data = {
                'user_id': user_id,
                'user_name': user_name or '',
                'user_email': user_email or ''
            }
            response = requests.post(
                f"{self.base_url}/upload-user-pdf",
                files=files,
                data=data
            )
        return response.json()
    
    def query(
        self,
        question: str,
        user_id: str = None,
        use_user_pdf: bool = False
    ):
        """
        Query the RAG system
        
        Args:
            question: The question to ask
            user_id: Optional user identifier
            use_user_pdf: Whether to include user-uploaded PDFs
        """
        payload = {
            "question": question,
            "user_id": user_id,
            "use_user_pdf": use_user_pdf
        }
        response = requests.post(
            f"{self.base_url}/query",
            json=payload
        )
        return response.json()
    
    def query_base_only(self, question: str):
        """
        Query only the base PDFs
        
        Args:
            question: The question to ask
        """
        response = requests.post(
            f"{self.base_url}/query-base-only",
            data={"question": question}
        )
        return response.json()
    
    def list_base_pdfs(self):
        """List all base PDFs"""
        response = requests.get(f"{self.base_url}/list-base-pdfs")
        return response.json()
    
    def list_user_pdfs(self, user_id: str):
        """
        List PDFs for a specific user
        
        Args:
            user_id: User identifier
        """
        response = requests.get(f"{self.base_url}/list-user-pdfs/{user_id}")
        return response.json()
    
    def delete_user_data(self, user_id: str):
        """
        Delete all data for a user
        
        Args:
            user_id: User identifier
        """
        response = requests.delete(f"{self.base_url}/delete-user-data/{user_id}")
        return response.json()


# Example usage
if __name__ == "__main__":
    # Initialize client
    client = PDFRAGClient("http://localhost:8000")
    
    print("="*60)
    print("PDF RAG API Client - Example Usage")
    print("="*60)
    
    # 1. Health check
    print("\n1. Checking API health...")
    health = client.health_check()
    print(f"Status: {health['status']}")
    print(f"Base documents: {health['base_documents']}")
    
    # 2. List base PDFs
    print("\n2. Listing base PDFs...")
    base_pdfs = client.list_base_pdfs()
    print(f"Found {base_pdfs['count']} base PDFs:")
    for pdf in base_pdfs['pdfs']:
        print(f"  - {pdf}")
    
    # 3. Query base PDFs only
    print("\n3. Querying base PDFs...")
    question = "What is the main topic of the documents?"
    result = client.query_base_only(question)
    print(f"Question: {result['question']}")
    print(f"Answer: {result['answer']}")
    print(f"Sources: {len(result['sources'])} documents")
    
    # 4. Upload user PDF (example - uncomment if you have a PDF to upload)
    # print("\n4. Uploading user PDF...")
    # user_id = "user123"
    # upload_result = client.upload_user_pdf(
    #     pdf_path="path/to/user/document.pdf",
    #     user_id=user_id,
    #     user_name="John Doe",
    #     user_email="john@example.com"
    # )
    # print(f"Upload result: {upload_result['message']}")
    
    # 5. Query with user PDF (example)
    # print("\n5. Querying with user PDF...")
    # result = client.query(
    #     question="What information is in my document?",
    #     user_id=user_id,
    #     use_user_pdf=True
    # )
    # print(f"Answer: {result['answer']}")
    
    print("\n" + "="*60)
    print("Example completed!")
    print("="*60)
