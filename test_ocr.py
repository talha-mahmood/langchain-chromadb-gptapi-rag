"""
Test OCR functionality with scanned PDFs
"""

from pathlib import Path
from vris_rag_system import inspect_pdf_text_coverage, VRISRAGSystem

def test_ocr_on_pdfs():
    """Test OCR on PDFs in user_pdfs folder"""
    
    # Find all PDFs
    pdf_folder = Path("user_pdfs/1")
    pdfs = list(pdf_folder.glob("*.pdf"))
    
    if not pdfs:
        print("No PDFs found in user_pdfs/1/")
        return
    
    print(f"\n{'='*70}")
    print("Testing OCR Detection")
    print(f"{'='*70}\n")
    
    for pdf in pdfs:
        print(f"📄 {pdf.name}")
        stats = inspect_pdf_text_coverage(str(pdf))

        if stats.get("is_mixed"):
            print(
                "   ✓ MIXED PDF - Will use OCR "
                f"({stats.get('text_pages', 0)}/{stats.get('total_pages', 0)} pages have extractable text)"
            )
        elif stats.get("is_scanned"):
            print("   ✓ SCANNED PDF - Will use OCR")
        else:
            print("   ✓ TEXT-BASED PDF - Will use standard extraction")
        print()
    
    print(f"\n{'='*70}")
    print("Processing PDFs with VRIS")
    print(f"{'='*70}\n")
    
    # Initialize VRIS
    vris = VRISRAGSystem(
        system_docs_folder="./system-doc",
        veteran_docs_folder="",
        persist_directory="./chroma_db",
        model_name="gpt-4"
    )
    
    # Load system vectorstore
    vris.create_system_vectorstore(force_reload=False)
    
    # Process veteran documents
    pdf_paths = [str(pdf) for pdf in pdfs]
    vris.process_veteran_documents_from_upload(pdf_paths)
    
    print(f"\n{'='*70}")
    print("OCR Results")
    print(f"{'='*70}")
    
    ocr_output_dir = Path("ocr_output")
    if ocr_output_dir.exists():
        ocr_files = list(ocr_output_dir.glob("*.txt"))
        if ocr_files:
            print(f"\n✓ {len(ocr_files)} OCR text file(s) saved in 'ocr_output/' folder:")
            for ocr_file in ocr_files:
                size_kb = ocr_file.stat().st_size / 1024
                print(f"  • {ocr_file.name} ({size_kb:.1f} KB)")
            print("\nYou can review these files to verify OCR accuracy!")
        else:
            print("\nNo scanned PDFs detected (all were text-based)")
    else:
        print("\nNo scanned PDFs detected (all were text-based)")
    
    print(f"\n{'='*70}\n")

if __name__ == "__main__":
    test_ocr_on_pdfs()
