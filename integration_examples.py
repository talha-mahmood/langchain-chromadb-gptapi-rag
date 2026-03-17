"""
Example: How to integrate PDF RAG API from another Python project.
This shows real-world usage patterns.
"""

import requests
import json
from typing import Optional


class PDFRAGIntegration:
    """
    Integration class for PDF RAG API.
    Use this in your external project to interact with the RAG system.
    """
    
    def __init__(self, api_base_url: str = "http://localhost:8000"):
        self.api_base_url = api_base_url
        self.session = requests.Session()
    
    def upload_user_document(
        self,
        file_path: str,
        user_id: str,
        user_metadata: Optional[dict] = None
    ) -> dict:
        """
        Upload a user's document to the RAG system.
        
        Args:
            file_path: Path to the PDF file
            user_id: Unique identifier for the user
            user_metadata: Optional metadata (name, email, etc.)
        
        Returns:
            Response from the API
        """
        url = f"{self.api_base_url}/upload-user-pdf"
        
        with open(file_path, 'rb') as f:
            files = {'file': f}
            data = {'user_id': user_id}
            
            if user_metadata:
                if 'name' in user_metadata:
                    data['user_name'] = user_metadata['name']
                if 'email' in user_metadata:
                    data['user_email'] = user_metadata['email']
            
            response = self.session.post(url, files=files, data=data)
            response.raise_for_status()
            return response.json()
    
    def ask_question(
        self,
        question: str,
        user_id: Optional[str] = None,
        include_user_documents: bool = False
    ) -> dict:
        """
        Ask a question to the RAG system.
        
        Args:
            question: The question to ask
            user_id: User ID (required if include_user_documents is True)
            include_user_documents: Whether to search user's uploaded documents
        
        Returns:
            Answer and sources
        """
        url = f"{self.api_base_url}/query"
        
        payload = {
            "question": question,
            "user_id": user_id,
            "use_user_pdf": include_user_documents
        }
        
        response = self.session.post(url, json=payload)
        response.raise_for_status()
        return response.json()
    
    def get_base_documents_info(self) -> dict:
        """Get information about the base knowledge documents."""
        response = self.session.get(f"{self.api_base_url}/list-base-pdfs")
        response.raise_for_status()
        return response.json()
    
    def cleanup_user_data(self, user_id: str) -> dict:
        """
        Delete all data for a user (e.g., when user account is deleted).
        
        Args:
            user_id: User identifier
        """
        url = f"{self.api_base_url}/delete-user-data/{user_id}"
        response = self.session.delete(url)
        response.raise_for_status()
        return response.json()


# ============================================================================
# EXAMPLE 1: Flask Application Integration
# ============================================================================

def flask_integration_example():
    """
    Example of how to integrate with a Flask application.
    """
    from flask import Flask, request, jsonify
    
    app = Flask(__name__)
    rag_client = PDFRAGIntegration("http://localhost:8000")
    
    @app.route('/user/upload-document', methods=['POST'])
    def upload_document():
        """Endpoint for users to upload their documents"""
        file = request.files['file']
        user_id = request.form['user_id']
        
        # Save temporarily
        temp_path = f"/tmp/{file.filename}"
        file.save(temp_path)
        
        # Upload to RAG system
        result = rag_client.upload_user_document(
            file_path=temp_path,
            user_id=user_id,
            user_metadata={
                'name': request.form.get('name'),
                'email': request.form.get('email')
            }
        )
        
        return jsonify(result)
    
    @app.route('/ask', methods=['POST'])
    def ask_question():
        """Endpoint to ask questions"""
        data = request.json
        
        result = rag_client.ask_question(
            question=data['question'],
            user_id=data.get('user_id'),
            include_user_documents=data.get('include_user_docs', False)
        )
        
        return jsonify({
            'answer': result['answer'],
            'sources': result['sources']
        })
    
    return app


# ============================================================================
# EXAMPLE 2: Django View Integration
# ============================================================================

def django_integration_example():
    """
    Example of how to integrate with Django views.
    """
    from django.http import JsonResponse
    from django.views.decorators.http import require_http_methods
    
    rag_client = PDFRAGIntegration("http://localhost:8000")
    
    @require_http_methods(["POST"])
    def upload_user_document(request):
        """Django view to upload user document"""
        file = request.FILES['document']
        user_id = request.POST['user_id']
        
        # Save to temp location
        temp_path = f"/tmp/{file.name}"
        with open(temp_path, 'wb+') as destination:
            for chunk in file.chunks():
                destination.write(chunk)
        
        # Upload to RAG
        result = rag_client.upload_user_document(
            file_path=temp_path,
            user_id=user_id,
            user_metadata={
                'name': request.POST.get('name'),
                'email': request.user.email
            }
        )
        
        return JsonResponse(result)
    
    @require_http_methods(["POST"])
    def ask_rag_question(request):
        """Django view to query RAG system"""
        data = json.loads(request.body)
        
        result = rag_client.ask_question(
            question=data['question'],
            user_id=str(request.user.id),
            include_user_documents=data.get('include_user_docs', True)
        )
        
        return JsonResponse(result)


# ============================================================================
# EXAMPLE 3: Simple Script Integration
# ============================================================================

def simple_script_example():
    """
    Simple standalone script example.
    """
    # Initialize
    rag = PDFRAGIntegration("http://localhost:8000")
    
    # User workflow
    user_id = "user_12345"
    
    # Step 1: Upload user's document
    print("Uploading user document...")
    upload_result = rag.upload_user_document(
        file_path="./user_document.pdf",
        user_id=user_id,
        user_metadata={
            'name': 'John Doe',
            'email': 'john@example.com'
        }
    )
    print(f"✓ Uploaded: {upload_result['filename']}")
    
    # Step 2: Ask question using base knowledge only
    print("\nQuerying base knowledge...")
    answer1 = rag.ask_question(
        question="What are the company policies?",
        include_user_documents=False
    )
    print(f"Answer: {answer1['answer']}")
    
    # Step 3: Ask question including user's document
    print("\nQuerying with user document...")
    answer2 = rag.ask_question(
        question="What information is in my document?",
        user_id=user_id,
        include_user_documents=True
    )
    print(f"Answer: {answer2['answer']}")
    print(f"Sources: {len(answer2['sources'])} documents")
    
    # Step 4: Cleanup when done
    print("\nCleaning up user data...")
    rag.cleanup_user_data(user_id)
    print("✓ User data deleted")


# ============================================================================
# EXAMPLE 4: Async Integration (for async frameworks)
# ============================================================================

async def async_integration_example():
    """
    Example using aiohttp for async applications.
    """
    import aiohttp
    
    async def upload_and_query(user_id: str, pdf_path: str, question: str):
        async with aiohttp.ClientSession() as session:
            # Upload PDF
            with open(pdf_path, 'rb') as f:
                data = aiohttp.FormData()
                data.add_field('file', f, filename='document.pdf')
                data.add_field('user_id', user_id)
                
                async with session.post(
                    'http://localhost:8000/upload-user-pdf',
                    data=data
                ) as resp:
                    upload_result = await resp.json()
                    print(f"Uploaded: {upload_result['message']}")
            
            # Query
            payload = {
                'question': question,
                'user_id': user_id,
                'use_user_pdf': True
            }
            
            async with session.post(
                'http://localhost:8000/query',
                json=payload
            ) as resp:
                query_result = await resp.json()
                return query_result
    
    # Usage
    result = await upload_and_query(
        user_id='user123',
        pdf_path='document.pdf',
        question='What is in this document?'
    )
    print(result['answer'])


# ============================================================================
# EXAMPLE 5: Batch Processing
# ============================================================================

def batch_processing_example():
    """
    Process multiple users and documents in batch.
    """
    rag = PDFRAGIntegration("http://localhost:8000")
    
    # Multiple users with their documents
    users = [
        {'id': 'user1', 'name': 'Alice', 'doc': 'alice_report.pdf'},
        {'id': 'user2', 'name': 'Bob', 'doc': 'bob_analysis.pdf'},
        {'id': 'user3', 'name': 'Charlie', 'doc': 'charlie_data.pdf'},
    ]
    
    # Upload all documents
    print("Uploading documents...")
    for user in users:
        result = rag.upload_user_document(
            file_path=user['doc'],
            user_id=user['id'],
            user_metadata={'name': user['name']}
        )
        print(f"✓ {user['name']}: {result['message']}")
    
    # Ask same question to all users
    question = "What are the key findings in your document?"
    print(f"\nAsking: {question}")
    
    for user in users:
        answer = rag.ask_question(
            question=question,
            user_id=user['id'],
            include_user_documents=True
        )
        print(f"\n{user['name']}: {answer['answer'][:100]}...")


# ============================================================================
# Main execution
# ============================================================================

if __name__ == "__main__":
    print("PDF RAG API Integration Examples")
    print("="*60)
    
    # Run simple example
    print("\nRunning simple script example...")
    try:
        simple_script_example()
    except Exception as e:
        print(f"Error: {e}")
        print("Make sure the API is running at http://localhost:8000")
