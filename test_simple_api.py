"""
Test script for VRIS Simple API
Demonstrates how to call the single endpoint
"""

import requests
import json
from pathlib import Path

# API endpoint
API_URL = "http://localhost:8000/api/vris/analyze"

# Veteran documents to upload
DOCUMENTS = [
    "./user_pdfs/1/sample_va_decision_letter.txt",
    "./user_pdfs/1/sample_cp_exam_back.txt",
    "./user_pdfs/1/sample_cp_exam_ptsd.txt",
    "./user_pdfs/1/sample_private_sleep_study.txt"
]

def test_vris_api():
    """Test the VRIS API with sample documents"""
    
    print("\n" + "="*70)
    print("Testing VRIS™ Simple API")
    print("="*70 + "\n")
    
    # Prepare files for upload
    files = []
    for doc_path in DOCUMENTS:
        if Path(doc_path).exists():
            files.append(
                ('files', (Path(doc_path).name, open(doc_path, 'rb'), 'text/plain'))
            )
            print(f"📄 Uploading: {Path(doc_path).name}")
        else:
            print(f"⚠️  File not found: {doc_path}")
    
    if not files:
        print("❌ No files to upload")
        return
    
    print(f"\n🚀 Sending {len(files)} document(s) to VRIS API...")
    print("⏳ This may take 30-60 seconds...\n")
    
    try:
        # Call API
        response = requests.post(API_URL, files=files)
        
        # Close file handles
        for _, (_, file_handle, _) in files:
            file_handle.close()
        
        # Check response
        if response.status_code == 200:
            result = response.json()
            
            print("✅ Analysis Complete!\n")
            print("="*70)
            print("VRIS ANALYSIS RESULTS")
            print("="*70 + "\n")
            
            # Summary
            summary = result.get('summary', {})
            print("📊 SUMMARY:")
            print(f"   Total Opportunities: {summary.get('total_opportunities_identified', 0)}")
            print(f"   Underrated Conditions: {summary.get('underrated_conditions_count', 0)}")
            print(f"   Missed Conditions: {summary.get('missed_conditions_count', 0)}")
            print(f"   Secondary Conditions: {summary.get('secondary_conditions_count', 0)}")
            print(f"   High Confidence (90%+): {summary.get('high_confidence_opportunities', 0)}")
            print(f"   Recommendation: {summary.get('recommendation', 'N/A')}")
            
            # Findings
            findings = result.get('findings', {})
            
            if findings.get('underrated_conditions'):
                print("\n🔍 UNDERRATED CONDITIONS:")
                for cond in findings['underrated_conditions']:
                    print(f"\n   • {cond['name']}")
                    print(f"     Current: {cond.get('current_rating', 'N/A')}")
                    print(f"     Potential: {cond.get('potential_rating', 'N/A')}")
                    print(f"     Confidence: {cond.get('confidence', 'N/A')}%")
                    print(f"     Phase: {cond.get('phase', 'N/A')}")
            
            if findings.get('missed_conditions'):
                print("\n❌ MISSED CONDITIONS (Not Currently Rated):")
                for cond in findings['missed_conditions']:
                    print(f"\n   • {cond['name']}")
                    print(f"     Potential Rating: {cond.get('potential_rating', 'N/A')}")
                    print(f"     Confidence: {cond.get('confidence', 'N/A')}%")
                    print(f"     Phase: {cond.get('phase', 'N/A')}")
            
            if findings.get('secondary_conditions'):
                print("\n🔗 SECONDARY CONDITIONS:")
                for cond in findings['secondary_conditions']:
                    print(f"\n   • {cond['name']}")
                    print(f"     Confidence: {cond.get('confidence', 'N/A')}%")
            
            # Full response (optional - comment out for cleaner output)
            print("\n" + "="*70)
            print("FULL API RESPONSE:")
            print("="*70)
            print(json.dumps(result, indent=2))
            
        else:
            print(f"❌ API Error: {response.status_code}")
            print(response.text)
    
    except requests.exceptions.ConnectionError:
        print("❌ Connection Error: Is the API server running?")
        print("   Start it with: python vris_simple_api.py")
    except Exception as e:
        print(f"❌ Error: {e}")


def test_health_check():
    """Test the health check endpoint"""
    try:
        response = requests.get("http://localhost:8000/api/vris/health")
        if response.status_code == 200:
            print("✅ API is healthy")
            print(json.dumps(response.json(), indent=2))
        else:
            print(f"⚠️  API returned status {response.status_code}")
    except:
        print("❌ API is not running")


if __name__ == "__main__":
    # First check if API is running
    print("Checking API health...")
    test_health_check()
    print()
    
    # Then test the main analysis endpoint
    test_vris_api()
