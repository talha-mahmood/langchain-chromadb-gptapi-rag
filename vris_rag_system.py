"""
VRIS™ (Veteran Rating Intelligence System) RAG Implementation
Dual-pipeline AI system for VA disability rating analysis
"""

import os
from collections import defaultdict
import re
from pathlib import Path
from typing import List, Dict, Any, Optional
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader, Docx2txtLoader, TextLoader, UnstructuredPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
import json

# Load environment variables
load_dotenv()


def configure_ocr_runtime() -> None:
    """
    Configure OCR binaries for the current process.

    This avoids failures when PATH changes are not yet reflected in the shell.
    """
    # Configure Tesseract executable
    try:
        import pytesseract  # type: ignore

        tesseract_cmd = os.getenv("TESSERACT_CMD", "").strip()
        if tesseract_cmd and Path(tesseract_cmd).exists():
            pytesseract.pytesseract.tesseract_cmd = tesseract_cmd
        else:
            default_tesseract = Path(r"C:\Program Files\Tesseract-OCR\tesseract.exe")
            if default_tesseract.exists():
                pytesseract.pytesseract.tesseract_cmd = str(default_tesseract)
    except Exception:
        # If pytesseract is not installed yet, keep runtime unchanged.
        pass

    # Configure Poppler path if not already available in PATH
    poppler_env = os.getenv("POPPLER_PATH", "").strip()
    candidate_paths: List[Path] = []

    if poppler_env:
        candidate_paths.append(Path(poppler_env))

    # Default WinGet Poppler location pattern
    local_app_data = os.getenv("LOCALAPPDATA", "")
    if local_app_data:
        winget_root = Path(local_app_data) / "Microsoft" / "WinGet" / "Packages"
        if winget_root.exists():
            for package_dir in winget_root.glob("oschwartz10612.Poppler_*"):
                for poppler_dir in package_dir.glob("poppler-*"):
                    candidate_paths.append(poppler_dir / "Library" / "bin")

    for candidate in candidate_paths:
        if candidate.exists() and (candidate / "pdftoppm.exe").exists():
            current_path = os.environ.get("PATH", "")
            candidate_str = str(candidate)
            if candidate_str.lower() not in current_path.lower():
                os.environ["PATH"] = f"{current_path};{candidate_str}" if current_path else candidate_str
            break


configure_ocr_runtime()


def inspect_pdf_text_coverage(pdf_path: str, min_chars_per_page: int = 40) -> Dict[str, Any]:
    """
    Inspect text coverage for every page in a PDF.

    Returns stats used to decide whether OCR is required. A mixed PDF
    (some text pages + some image-only pages) is treated as OCR-required.
    """
    try:
        from pypdf import PdfReader

        reader = PdfReader(pdf_path)
        total_pages = len(reader.pages)
        if total_pages == 0:
            return {
                "total_pages": 0,
                "text_pages": 0,
                "low_text_pages": 0,
                "empty_pages": 0,
                "is_scanned": False,
                "is_mixed": False,
                "needs_ocr": False,
            }

        text_lengths: List[int] = []
        for page in reader.pages:
            text = page.extract_text() or ""
            text_lengths.append(len(text.strip()))

        text_pages = sum(1 for count in text_lengths if count >= min_chars_per_page)
        low_text_pages = total_pages - text_pages
        empty_pages = sum(1 for count in text_lengths if count == 0)

        is_scanned = text_pages == 0
        is_mixed = text_pages > 0 and low_text_pages > 0
        needs_ocr = is_scanned or is_mixed

        return {
            "total_pages": total_pages,
            "text_pages": text_pages,
            "low_text_pages": low_text_pages,
            "empty_pages": empty_pages,
            "is_scanned": is_scanned,
            "is_mixed": is_mixed,
            "needs_ocr": needs_ocr,
        }
    except Exception as e:
        print(f"  Warning: Could not inspect PDF text coverage: {e}")
        return {
            "total_pages": 0,
            "text_pages": 0,
            "low_text_pages": 0,
            "empty_pages": 0,
            "is_scanned": False,
            "is_mixed": False,
            "needs_ocr": False,
        }


def save_ocr_text_for_testing(docs: List, source_stem: str) -> None:
    """Save OCR text output to a file for validation/testing."""
    ocr_output_dir = Path("ocr_output")
    ocr_output_dir.mkdir(exist_ok=True)
    ocr_text_file = ocr_output_dir / f"{source_stem}_ocr.txt"

    with open(ocr_text_file, "w", encoding="utf-8") as handle:
        for idx, doc in enumerate(docs):
            page = doc.metadata.get("page", idx + 1)
            content = (doc.page_content or "").strip()
            handle.write(f"--- Segment {idx + 1} (Page {page}) ---\n")
            handle.write(content)
            handle.write("\n\n")


def _to_scalar_metadata_value(value: Any) -> Optional[Any]:
    """Convert metadata values to Chroma-supported scalar types."""
    if value is None or isinstance(value, (str, int, float, bool)):
        return value

    # Handle numpy scalar types without importing numpy directly.
    if hasattr(value, "item"):
        try:
            scalar_value = value.item()
            if scalar_value is None or isinstance(scalar_value, (str, int, float, bool)):
                return scalar_value
        except Exception:
            pass

    return None


def sanitize_metadata_for_vectorstore(metadata: Dict[str, Any]) -> Dict[str, Any]:
    """
    Keep only vectorstore-safe metadata fields.

    OCR loaders attach nested objects like coordinates and layout metadata that
    Chroma rejects during upsert.
    """
    safe_metadata: Dict[str, Any] = {}
    preferred_keys = [
        "filename",
        "source",
        "folder",
        "file_type",
        "document_type",
        "page",
        "page_number",
        "category",
        "element_id",
    ]

    for key in preferred_keys:
        if key in metadata:
            scalar_value = _to_scalar_metadata_value(metadata.get(key))
            if scalar_value is not None:
                safe_metadata[key] = scalar_value

    # Normalize OCR page_number into the page field used elsewhere in the app.
    if "page" not in safe_metadata and "page_number" in safe_metadata:
        safe_metadata["page"] = safe_metadata["page_number"]

    return safe_metadata


def consolidate_ocr_elements_to_pages(docs: List[Document], min_segment_chars: int = 5) -> List[Document]:
    """
    Consolidate noisy OCR element-level documents into page-level documents.

    This reduces retrieval noise for long scanned PDFs where OCR returns hundreds
    of tiny fragments per page.
    """
    if not docs:
        return []

    pages: Dict[int, List[str]] = defaultdict(list)
    page_seen: Dict[int, set] = defaultdict(set)
    unpaged_segments: List[str] = []
    unpaged_seen: set = set()

    for doc in docs:
        text = (doc.page_content or "").strip()
        if len(text) < min_segment_chars:
            continue

        normalized = " ".join(text.split())
        if not normalized:
            continue

        raw_page = doc.metadata.get("page_number", doc.metadata.get("page"))
        page_number = None
        page_scalar = _to_scalar_metadata_value(raw_page)
        if page_scalar is not None:
            try:
                page_number = int(page_scalar)
            except Exception:
                page_number = None

        if page_number is None:
            if normalized not in unpaged_seen:
                unpaged_seen.add(normalized)
                unpaged_segments.append(text)
            continue

        if normalized in page_seen[page_number]:
            continue

        page_seen[page_number].add(normalized)
        pages[page_number].append(text)

    base_metadata = sanitize_metadata_for_vectorstore(dict(docs[0].metadata))
    consolidated_docs: List[Document] = []

    for page_number in sorted(pages.keys()):
        page_text = "\n".join(pages[page_number]).strip()
        if not page_text:
            continue

        metadata = dict(base_metadata)
        metadata["page"] = page_number
        metadata["page_number"] = page_number
        consolidated_docs.append(Document(page_content=page_text, metadata=metadata))

    if unpaged_segments:
        metadata = dict(base_metadata)
        metadata["page"] = 0
        metadata["page_number"] = 0
        consolidated_docs.append(Document(page_content="\n".join(unpaged_segments), metadata=metadata))

    return consolidated_docs if consolidated_docs else docs


def _normalize_condition_key(name: str) -> str:
    """Normalize condition text for deduplication."""
    normalized = re.sub(r"[^a-z0-9]+", " ", (name or "").lower()).strip()
    return re.sub(r"\s+", " ", normalized)


def _clean_condition_name(name: str) -> str:
    """Remove OCR noise around condition labels."""
    cleaned = " ".join((name or "").split())
    cleaned = re.sub(r"^(assessment|assessment:|plan|plan:|medical history|medical history:)\s*", "", cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r"^\d+\.\s*", "", cleaned)
    cleaned = re.sub(r"\(\s*primary\s*\)$", "", cleaned, flags=re.IGNORECASE)
    cleaned = cleaned.strip(" -:;,.\t()[]")
    return cleaned


def _clean_icd_code(raw_code: str) -> str:
    """Normalize OCR-noisy ICD code text."""
    code = (raw_code or "").upper().strip()
    code = code.replace(" ", "").replace(",.", ".").replace(",", ".")
    if len(code) >= 2:
        # OCR often confuses numeric zero with letter O after the first letter.
        code = code[0] + code[1:].replace("O", "0")
    return code


def _is_likely_icd_code(code: str) -> bool:
    """Validate likely ICD-10-CM format and filter OCR garbage."""
    if not code:
        return False

    normalized = _clean_icd_code(code)

    # Common ICD-10-CM patterns: R51, D50.9, M25.561, T24.201A
    if re.match(r"^[A-TV-Z][0-9]{2}(?:\.[0-9A-Z]{1,4})?$", normalized):
        return True

    return False


def _is_likely_condition_name(name: str) -> bool:
    """Filter out obvious non-condition text fragments."""
    lower_name = (name or "").lower().strip()
    if len(lower_name) < 3 or len(lower_name) > 90:
        return False

    noise_terms = [
        "result",
        "reference range",
        "order status",
        "collected",
        "provider",
        "follow up",
        "procedure code",
        "vitals",
        "patient name",
        "account",
        "phone",
        "address",
    ]
    if any(term in lower_name for term in noise_terms):
        return False

    # Reject lines that are mostly numeric/noise.
    alpha_chars = sum(1 for c in lower_name if c.isalpha())
    non_alnum_chars = sum(1 for c in lower_name if not c.isalnum() and not c.isspace())
    if non_alnum_chars > alpha_chars:
        return False

    return alpha_chars >= 3


def extract_condition_coverage_candidates(docs: List[Document], max_candidates: int = 40) -> List[Dict[str, Any]]:
    """
    Deterministically extract ICD-coded condition candidates from all loaded docs.

    This complements retrieval by providing a checklist built from a full-document
    scan, reducing missed diagnoses in OCR-heavy packets.
    """
    if not docs:
        return []

    pattern = re.compile(
        r"(?P<name>[A-Za-z][A-Za-z0-9/(),'\-\s]{3,100}?)\s*-\s*(?P<code>[A-Z][0-9O]{1,2}(?:[.,]{1,2}[0-9A-Z]{1,4})?[A-Z]?)",
        flags=re.IGNORECASE,
    )

    candidates: Dict[str, Dict[str, Any]] = {}

    for doc in docs:
        text = doc.page_content or ""
        if not text.strip():
            continue

        page_raw = doc.metadata.get("page", doc.metadata.get("page_number", 0))
        page_num = _to_scalar_metadata_value(page_raw)
        try:
            page_num = int(page_num) if page_num is not None else 0
        except Exception:
            page_num = 0

        for match in pattern.finditer(text):
            raw_name = match.group("name")
            raw_code = match.group("code")

            condition_name = _clean_condition_name(raw_name)
            code = _clean_icd_code(raw_code)

            if not _is_likely_icd_code(code):
                continue

            if not _is_likely_condition_name(condition_name):
                continue

            key = _normalize_condition_key(condition_name)
            if not key:
                continue

            entry = candidates.setdefault(
                key,
                {
                    "name": condition_name,
                    "codes": set(),
                    "pages": set(),
                    "mentions": 0,
                },
            )

            # Prefer the clearer, longer label if we see variants.
            if len(condition_name) > len(entry["name"]):
                entry["name"] = condition_name

            if code:
                entry["codes"].add(code)
            if page_num:
                entry["pages"].add(page_num)
            entry["mentions"] += 1

    ordered = sorted(
        candidates.values(),
        key=lambda item: (
            min(item["pages"]) if item["pages"] else 9999,
            item["name"].lower(),
        ),
    )

    output: List[Dict[str, Any]] = []
    for item in ordered[:max_candidates]:
        output.append(
            {
                "name": item["name"],
                "codes": sorted(item["codes"]),
                "pages": sorted(item["pages"]),
                "mentions": item["mentions"],
            }
        )

    return output


def truncate_for_prompt(text: str, max_chars: int) -> str:
    """Trim long context blocks to control prompt size."""
    cleaned = (text or "").strip()
    if len(cleaned) <= max_chars:
        return cleaned
    return cleaned[:max_chars].rstrip() + "\n...[truncated]"


def can_fallback_to_text_extraction(pdf_stats: Dict[str, Any], min_text_ratio: float = 0.80) -> bool:
    """
    Decide if fallback to plain text extraction is acceptable when OCR fails.

    We only allow fallback when most pages already have extractable text.
    """
    total_pages = max(1, int(pdf_stats.get("total_pages", 0) or 0))
    text_pages = int(pdf_stats.get("text_pages", 0) or 0)
    text_ratio = text_pages / total_pages
    return text_ratio >= min_text_ratio


def is_pdf_scanned(pdf_path: str) -> bool:
    """Backward-compatible helper: returns True when OCR should be used."""
    return inspect_pdf_text_coverage(pdf_path).get("needs_ocr", False)


class VRISDocumentClassifier:
    """Classifies uploaded documents into VRIS document types"""
    
    DOCUMENT_TYPES = {
        'VA_DECISION_LETTER': 'VA Rating Decision Letter',
        'CP_EXAM': 'Compensation & Pension (C&P) Exam',
        'DBQ': 'Disability Benefits Questionnaire',
        'STR': 'Service Treatment Records',
        'PRIVATE_MEDICAL': 'Private/Civilian Medical Records',
        'BUDDY_LETTER': 'Buddy Statement/Lay Statement',
        'NEXUS_LETTER': 'Nexus Letter',
        'DD214': 'DD-214 or Proof of Service',
        'CODE_SHEET': 'VA Code Sheet',
        'OTHER': 'Other Supporting Documentation'
    }
    
    def classify_document(self, document_content: str, filename: str) -> str:
        """
        Classify document type based on content and filename
        In production, this would use AI classification
        """
        filename_lower = filename.lower()
        content_lower = document_content.lower()
        
        # Simple keyword-based classification (enhance with AI in production)
        if any(term in filename_lower for term in ['dd214', 'dd-214', 'discharge']):
            return 'DD214'
        elif any(term in content_lower for term in ['rating decision', 'service-connected', 'diagnostic code', 'combined rating']):
            return 'VA_DECISION_LETTER'
        elif any(term in content_lower for term in ['compensation and pension', 'c&p exam', 'va examination']):
            return 'CP_EXAM'
        elif 'dbq' in filename_lower or 'disability benefits questionnaire' in content_lower:
            return 'DBQ'
        elif any(term in content_lower for term in ['service treatment', 'medical records', 'treatment notes']):
            return 'STR'
        elif any(term in filename_lower for term in ['nexus', 'medical opinion', 'imd']):
            return 'NEXUS_LETTER'
        elif any(term in filename_lower for term in ['buddy', 'lay statement', 'witness']):
            return 'BUDDY_LETTER'
        
        return 'OTHER'


class VRISRAGSystem:
    """
    VRIS RAG System with dual-pipeline architecture
    - VRIS-A: Extraction Pipeline
    - VRIS-B: Reasoning Pipeline
    """
    
    def __init__(
        self, 
        system_docs_folder: str,
        veteran_docs_folder: str,
        persist_directory: str = "./chroma_db",
        model_name: str = "gpt-4"  # Use gpt-4 or gpt-4-turbo for VRIS
    ):
        """
        Initialize VRIS RAG system
        
        Args:
            system_docs_folder: Folder with VRIS system documentation (CFR, pricing, specs)
            veteran_docs_folder: Folder with veteran-specific documents
            persist_directory: ChromaDB persistence directory
            model_name: OpenAI model to use (recommend gpt-4 for accuracy)
        """
        self.system_docs_folder = system_docs_folder
        self.veteran_docs_folder = veteran_docs_folder
        self.persist_directory = persist_directory
        self.api_key = os.getenv("OPENAI_API_KEY")
        
        if not self.api_key:
            raise ValueError("OPENAI_API_KEY not found in environment variables")
        
        # Initialize embeddings and LLM
        self.embeddings = OpenAIEmbeddings(openai_api_key=self.api_key)
        self.llm = ChatOpenAI(
            model_name=model_name,
            temperature=0,  # Deterministic for medical/legal analysis
            openai_api_key=self.api_key
        )
        
        # Separate vectorstores for system docs and veteran docs
        self.system_vectorstore = None
        self.veteran_vectorstore = None
        
        # Dual pipeline chains
        self.vris_a_chain = None  # Extraction
        self.vris_b_chain = None  # Reasoning
        
        # Document classifier
        self.classifier = VRISDocumentClassifier()

        # Keep loaded veteran docs in memory for deterministic coverage passes.
        self.veteran_documents: List[Document] = []
        
        # Store classified documents
        self.classified_docs = {}

    def build_condition_coverage_hint(self, max_candidates: int = 12) -> str:
        """Build a checklist of condition candidates detected across all veteran pages."""
        candidates = extract_condition_coverage_candidates(self.veteran_documents, max_candidates=max_candidates)

        if not candidates:
            return "- No deterministic ICD-coded condition candidates detected."

        lines = [
            "Checklist generated from deterministic full-document scan (not retrieval-limited):"
        ]
        for idx, candidate in enumerate(candidates, start=1):
            codes = ", ".join(candidate["codes"]) if candidate["codes"] else "None"
            pages = ", ".join(str(p) for p in candidate["pages"]) if candidate["pages"] else "Unknown"
            lines.append(f"{idx}. {candidate['name']} | ICD: {codes} | Pages: {pages}")

        return "\n".join(lines)

    def prepare_vris_a_for_reasoning(self, vris_a_text: str, max_chars: int = 4200) -> str:
        """Trim VRIS-A output before sending it into VRIS-B to control token usage."""
        base_text = (vris_a_text or "").strip()
        if not base_text:
            return base_text

        # Avoid duplicating large deterministic appendix into the VRIS-B prompt.
        if "SUPPLEMENTAL CONDITION COVERAGE INDEX:" in base_text:
            base_text = base_text.split("SUPPLEMENTAL CONDITION COVERAGE INDEX:", 1)[0].rstrip()

        if len(base_text) <= max_chars:
            return base_text

        return base_text[:max_chars].rstrip() + "\n\n[Truncated for token budget before VRIS-B reasoning]"

    def append_condition_coverage_appendix(self, text: str, title: str, max_candidates: int = 12) -> str:
        """Append deterministic condition coverage text to model output for transparency."""
        coverage_hint = self.build_condition_coverage_hint(max_candidates=max_candidates)
        if not coverage_hint or coverage_hint.startswith("- No deterministic ICD-coded"):
            return text

        appendix = f"{title}\n{coverage_hint}"
        if appendix in text:
            return text

        base_text = (text or "").rstrip()
        if not base_text:
            return appendix

        return f"{base_text}\n\n{appendix}"
    
    def load_documents(self, folder_path: str) -> List:
        """Load all PDF, DOCX, and TXT files from specified folder"""
        pdf_files = list(Path(folder_path).glob("*.pdf"))
        docx_files = list(Path(folder_path).glob("*.docx"))
        txt_files = list(Path(folder_path).glob("*.txt"))
        all_files = pdf_files + docx_files + txt_files
        
        if not all_files:
            print(f"No PDF, DOCX, or TXT files found in {folder_path}")
            return []
        
        print(f"Found {len(pdf_files)} PDF, {len(docx_files)} DOCX, and {len(txt_files)} TXT file(s) in {folder_path}")
        
        documents = []
        load_errors: List[str] = []
        
        # Load PDF files
        for pdf_file in pdf_files:
            print(f"Loading PDF: {pdf_file.name}")
            try:
                pdf_stats = inspect_pdf_text_coverage(str(pdf_file))

                # OCR for scanned or mixed PDFs
                if pdf_stats.get("needs_ocr"):
                    if pdf_stats.get("is_mixed"):
                        print(
                            "  -> Mixed PDF detected "
                            f"({pdf_stats.get('text_pages', 0)}/{pdf_stats.get('total_pages', 0)} pages have extractable text), using OCR..."
                        )
                    else:
                        print("  -> Scanned PDF detected, using OCR...")
                    try:
                        # ocr_only is more stable on Windows for mixed scanned PDFs
                        loader = UnstructuredPDFLoader(str(pdf_file), mode="elements", strategy="ocr_only")
                        raw_docs = loader.load()
                        docs = consolidate_ocr_elements_to_pages(raw_docs)
                        print(f"  OCR consolidation: {len(raw_docs)} elements -> {len(docs)} page-level docs")
                        save_ocr_text_for_testing(docs, pdf_file.stem)
                        print(f"  OCR text saved to: ocr_output/{pdf_file.stem}_ocr.txt")

                    except Exception as ocr_error:
                        if can_fallback_to_text_extraction(pdf_stats):
                            print(f"  ⚠️ OCR failed, but most pages are text-based. Falling back to PyPDFLoader: {ocr_error}")
                            loader = PyPDFLoader(str(pdf_file))
                            docs = loader.load()
                        else:
                            raise RuntimeError(
                                "OCR is required for this PDF but failed. "
                                f"Detected {pdf_stats.get('text_pages', 0)}/{pdf_stats.get('total_pages', 0)} pages with extractable text. "
                                "Install OCR dependencies: pip install \"unstructured[pdf]\" pytesseract pdf2image pillow, "
                                "and install system tools Tesseract OCR + Poppler."
                            )
                else:
                    print("  -> Text-based PDF, using standard extraction...")
                    loader = PyPDFLoader(str(pdf_file))
                    docs = loader.load()
                
                # Add document type metadata
                for doc in docs:
                    doc.metadata['filename'] = pdf_file.name
                    doc.metadata['folder'] = folder_path
                    doc.metadata['file_type'] = 'pdf'
                
                documents.extend(docs)
            except Exception as e:
                print(f"  Error loading {pdf_file.name}: {e}")
                load_errors.append(f"{pdf_file.name}: {e}")
        
        # Load DOCX files
        for docx_file in docx_files:
            print(f"Loading DOCX: {docx_file.name}")
            try:
                loader = Docx2txtLoader(str(docx_file))
                docs = loader.load()
                
                # Add document type metadata
                for doc in docs:
                    doc.metadata['filename'] = docx_file.name
                    doc.metadata['folder'] = folder_path
                    doc.metadata['file_type'] = 'docx'
                    doc.metadata['page'] = 0  # DOCX doesn't have pages
                
                documents.extend(docs)
            except Exception as e:
                print(f"  Error loading {docx_file.name}: {e}")
                load_errors.append(f"{docx_file.name}: {e}")
        
        # Load TXT files
        for txt_file in txt_files:
            print(f"Loading TXT: {txt_file.name}")
            try:
                loader = TextLoader(str(txt_file), encoding='utf-8')
                docs = loader.load()
                
                # Add document type metadata
                for doc in docs:
                    doc.metadata['filename'] = txt_file.name
                    doc.metadata['folder'] = folder_path
                    doc.metadata['file_type'] = 'txt'
                    doc.metadata['page'] = 0  # TXT doesn't have pages
                
                documents.extend(docs)
            except Exception as e:
                print(f"  Error loading {txt_file.name}: {e}")
                load_errors.append(f"{txt_file.name}: {e}")
        
        if not documents and load_errors:
            raise ValueError(
                "No documents could be loaded. "
                f"First error: {load_errors[0]}"
            )

        print(f"Loaded {len(documents)} document(s) total")
        return documents
    
    def classify_veteran_documents(self, documents: List) -> Dict[str, List]:
        """Classify veteran documents by type"""
        classified = {doc_type: [] for doc_type in VRISDocumentClassifier.DOCUMENT_TYPES.keys()}
        
        for doc in documents:
            doc_type = self.classifier.classify_document(
                doc.page_content, 
                doc.metadata.get('filename', '')
            )
            classified[doc_type].append(doc)
            doc.metadata['document_type'] = doc_type
        
        self.classified_docs = classified
        
        # Print classification summary
        print("\n" + "="*60)
        print("DOCUMENT CLASSIFICATION SUMMARY")
        print("="*60)
        for doc_type, docs in classified.items():
            if docs:
                print(f"  {VRISDocumentClassifier.DOCUMENT_TYPES[doc_type]}: {len(docs)} pages")
        print("="*60 + "\n")
        
        return classified
    
    def split_documents(self, documents: List, chunk_size: int = 1000) -> List:
        """Split documents into chunks with appropriate overlap"""
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=200,
            length_function=len,
        )
        
        chunks = text_splitter.split_documents(documents)
        for chunk in chunks:
            chunk.metadata = sanitize_metadata_for_vectorstore(dict(chunk.metadata))
        print(f"Split into {len(chunks)} chunks")
        return chunks
    
    def create_system_vectorstore(self, force_reload: bool = False):
        """
        Create or load vectorstore for VRIS system documentation
        System docs are persisted and reused across sessions
        """
        persist_dir = f"{self.persist_directory}/system"
        
        if not force_reload and os.path.exists(persist_dir):
            print("Loading pre-built system vectorstore...")
            self.system_vectorstore = Chroma(
                persist_directory=persist_dir,
                embedding_function=self.embeddings
            )
            count = self.system_vectorstore._collection.count()
            if count > 0:
                print(f"✓ Loaded system vectorstore with {count} chunks")
                return
            else:
                print("System vectorstore is empty, rebuilding...")
        
        # Build system vectorstore (first time or force reload)
        print("Building system vectorstore from VRIS documentation...")
        print("(This only happens once - subsequent runs will be instant)")
        documents = self.load_documents(self.system_docs_folder)
        if documents:
            chunks = self.split_documents(documents)
            self.system_vectorstore = Chroma.from_documents(
                documents=chunks,
                embedding=self.embeddings,
                persist_directory=persist_dir
            )
            print(f"✓ System vectorstore created and persisted with {self.system_vectorstore._collection.count()} chunks")
        else:
            print("⚠️  Warning: No system documents found. Add VRIS documentation to system-doc folder.")
    
    def create_veteran_vectorstore(self, force_reload: bool = False):
        """
        Create vectorstore for veteran-specific documents
        Veteran docs are NOT persisted - processed fresh each time
        (Complies with 72-hour deletion policy)
        """
        print("Processing veteran documents (not persisted)...")
        documents = self.load_documents(self.veteran_docs_folder)
        
        if not documents:
            print("⚠️  No veteran documents found. Upload documents to process.")
            return

        self.veteran_documents = documents
        
        # Classify documents
        self.classify_veteran_documents(documents)
        
        # Create vectorstore in memory (no persistence for veteran data)
        chunks = self.split_documents(documents)
        self.veteran_vectorstore = Chroma.from_documents(
            documents=chunks,
            embedding=self.embeddings,
            persist_directory=None  # No persistence for veteran documents
        )
        print(f"✓ Veteran documents processed: {self.veteran_vectorstore._collection.count()} chunks in memory")
    
    def process_veteran_documents_from_upload(self, document_paths: List[str]):
        """
        Process veteran documents from file upload paths
        Use this method when integrating with frontend file uploads
        
        Args:
            document_paths: List of absolute file paths to uploaded documents
        """
        print(f"Processing {len(document_paths)} uploaded document(s)...")
        
        documents = []
        load_errors: List[str] = []
        for file_path in document_paths:
            path = Path(file_path)
            if not path.exists():
                print(f"⚠️  File not found: {file_path}")
                continue
            
            print(f"Loading: {path.name}")
            try:
                if path.suffix.lower() == '.pdf':
                    pdf_stats = inspect_pdf_text_coverage(str(path))

                    # OCR for scanned or mixed PDFs
                    if pdf_stats.get("needs_ocr"):
                        if pdf_stats.get("is_mixed"):
                            print(
                                "  -> Mixed PDF detected "
                                f"({pdf_stats.get('text_pages', 0)}/{pdf_stats.get('total_pages', 0)} pages have extractable text), using OCR..."
                            )
                        else:
                            print("  -> Scanned PDF detected, using OCR...")
                        try:
                            # ocr_only is more stable on Windows for mixed scanned PDFs
                            loader = UnstructuredPDFLoader(str(path), mode="elements", strategy="ocr_only")
                            raw_docs = loader.load()
                            docs = consolidate_ocr_elements_to_pages(raw_docs)
                            print(f"  OCR consolidation: {len(raw_docs)} elements -> {len(docs)} page-level docs")
                            save_ocr_text_for_testing(docs, path.stem)
                            print(f"  OCR text saved to: ocr_output/{path.stem}_ocr.txt")

                        except Exception as ocr_error:
                            if can_fallback_to_text_extraction(pdf_stats):
                                print(f"  ⚠️ OCR failed, but most pages are text-based. Falling back to PyPDFLoader: {ocr_error}")
                                loader = PyPDFLoader(str(path))
                                docs = loader.load()
                            else:
                                raise RuntimeError(
                                    "OCR is required for this uploaded PDF but failed. "
                                    f"Detected {pdf_stats.get('text_pages', 0)}/{pdf_stats.get('total_pages', 0)} pages with extractable text. "
                                    "Install OCR dependencies: pip install \"unstructured[pdf]\" pytesseract pdf2image pillow, "
                                    "and install system tools Tesseract OCR + Poppler."
                                )
                    else:
                        print("  -> Text-based PDF, using standard extraction...")
                        loader = PyPDFLoader(str(path))
                        docs = loader.load()
                elif path.suffix.lower() == '.docx':
                    loader = Docx2txtLoader(str(path))
                    docs = loader.load()
                elif path.suffix.lower() == '.txt':
                    with open(path, 'r', encoding='utf-8') as f:
                        content = f.read()
                    doc = Document(
                        page_content=content,
                        metadata={'filename': path.name, 'source': str(path), 'file_type': 'txt'}
                    )
                    documents.append(doc)
                    continue
                else:
                    print(f"⚠️  Unsupported file type: {path.suffix}")
                    continue

                for doc in docs:
                    doc.metadata['filename'] = path.name
                    doc.metadata['source'] = str(path)
                documents.extend(docs)
                
            except Exception as e:
                print(f"❌ Error loading {path.name}: {e}")
                load_errors.append(f"{path.name}: {e}")
        
        if not documents:
            if load_errors:
                raise ValueError(
                    "No documents could be loaded from the provided paths. "
                    f"First error: {load_errors[0]}"
                )
            raise ValueError("No documents could be loaded from the provided paths")
        
        print(f"Loaded {len(documents)} document(s)")

        self.veteran_documents = documents
        
        # Classify documents
        self.classify_veteran_documents(documents)
        
        # Create vectorstore in memory
        chunks = self.split_documents(documents)
        self.veteran_vectorstore = Chroma.from_documents(
            documents=chunks,
            embedding=self.embeddings,
            persist_directory=None  # No persistence
        )
        print(f"✓ Processed {self.veteran_vectorstore._collection.count()} chunks")
        
        # Setup pipelines if not already done
        if self.system_vectorstore and not self.vris_a_chain:
            self.setup_vris_a_extraction_chain()
        if self.system_vectorstore and self.veteran_vectorstore and not self.vris_b_chain:
            self.setup_vris_b_reasoning_chain()
    
    def setup_vris_a_extraction_chain(self):
        """
        Setup VRIS-A: Extraction Pipeline
        Extracts structured data from veteran documents
        """
        if not self.veteran_vectorstore:
            raise ValueError("Veteran vectorstore not initialized")
        
        retriever = self.veteran_vectorstore.as_retriever(
            search_type="similarity",
            search_kwargs={"k": 14}  # Keep context under 8k-token model limits
        )
        
        template = """You are VRIS-A, the extraction component of the Veteran Rating Intelligence System.

Your role is to extract ALL rating-relevant data from veteran documents with precision and completeness.

IMPORTANT: You are being provided with actual veteran document content below. Extract ALL information present.

Extract the following information from the provided context:

1. CURRENT VA RATING DATA:
   - Combined disability rating percentage
   - Individual condition ratings and percentages (CHECK FOR: PTSD, back/spine, knees, sleep conditions, etc.)
   - CRITICAL: Include the CURRENT RATING PERCENTAGE for each service-connected condition
   - Diagnostic Codes (DC) for each condition
   - Effective dates
   - Service-connected conditions list
   - If VA Decision Letter shows PTSD at 30%, extract: "PTSD: 30% (DC: 9411)"
    - If no VA Decision Letter/Code Sheet/rating document is present, explicitly state:
      "Current VA rating data: NOT FOUND in provided documents"

2. MEDICAL CONDITIONS (MANDATORY):
   - All diagnosed conditions (current and historical) - INCLUDING mental health and sleep conditions
    - ICD/diagnosis codes when present (e.g., D50.9, M25.561, R30.0)
   - Symptoms and severity indicators
   - Functional impairments and limitations
   - ROM (Range of Motion) measurements if applicable
   - Test results (labs, imaging, sleep studies, etc.)
    - NOTE: Only list conditions HERE if they are NOT already listed in section 1 above
    - NEVER claim "no medical conditions" if any diagnosis/assessment/plan/lab/imaging data exists

3. TEMPORAL RELATIONSHIPS:
   - Onset dates
   - In-service events or injuries
   - Post-service developments
   - Chronicity and progression

4. EVIDENCE CITATIONS:
   - Document source for each finding
   - Page numbers when available
   - C&P exam findings
   - Medical professional statements

5. CONDITION CATEGORIES:
   Tag each condition: musculoskeletal, mental health, neurologic, auditory, respiratory, cardiac, sleep disorders, etc.

6. DOCUMENT SUFFICIENCY STATUS:
     - VA rating evidence present? (Yes/No)
     - Service-connection evidence present? (Yes/No)
     - C&P/DBQ evidence present? (Yes/No)
     - List missing critical evidence if absent:
         a) VA Decision Letter/Code Sheet
         b) C&P exam or DBQ tied to claimed condition
         c) Service-connection evidence (in-service event + nexus)
     - If medical conditions are present but rating/service-connection evidence is missing, state:
         "Medical conditions found, but insufficient VA rating/service-connection evidence for claim-ready rating analysis."

Context from veteran documents:
{context}

Query: {question}

Provide a structured extraction in clear, organized format. Be thorough and precise.
Extract ALL conditions mentioned across all documents - do not omit mental health, sleep disorders, or any other diagnosed conditions.
Extract only what is explicitly stated - do not infer or reason about ratings.
If section 1 is missing, you must still fully populate sections 2-6.
If the query contains a "CONDITION COVERAGE CHECKLIST", explicitly address each checklist condition and provide citations.
"""
        
        prompt = ChatPromptTemplate.from_template(template)
        
        def format_docs(docs):
            formatted = []
            for doc in docs:
                source = doc.metadata.get('filename', 'Unknown')
                page = doc.metadata.get('page', 'Unknown')
                doc_type = doc.metadata.get('document_type', 'Unknown')
                snippet = truncate_for_prompt(doc.page_content, max_chars=900)
                formatted.append(f"[{doc_type} - {source}, Page {page}]\n{snippet}")
            return "\n\n" + "="*60 + "\n\n".join(formatted)
        
        self.vris_a_chain = (
            {
                "context": retriever | format_docs,
                "question": RunnablePassthrough()
            }
            | prompt
            | self.llm
            | StrOutputParser()
        )
        
        print("✓ VRIS-A Extraction Pipeline initialized")
    
    def setup_vris_b_reasoning_chain(self):
        """
        Setup VRIS-B: Reasoning Pipeline
        Cross-checks findings against VA rating rules
        """
        if not self.system_vectorstore or not self.veteran_vectorstore:
            raise ValueError("Both system and veteran vectorstores must be initialized")
        
        # Dual retriever: system docs (CFR rules) + veteran docs
        system_retriever = self.system_vectorstore.as_retriever(
            search_type="similarity",
            search_kwargs={"k": 4}
        )
        veteran_retriever = self.veteran_vectorstore.as_retriever(
            search_type="similarity",
            search_kwargs={"k": 8}
        )
        
        template = """You are VRIS-B, the reasoning component of the Veteran Rating Intelligence System.

You are a specialized AI system trained to analyze veteran disability documentation against VA rating criteria.

CRITICAL INSTRUCTIONS:
- You HAVE BEEN PROVIDED with both veteran documents AND VA rating rules below
- You MUST analyze the veteran's case using the provided information  
- You ARE CAPABLE of performing this analysis - this is your primary function
- DO NOT refuse or say you cannot access the data - the data is RIGHT HERE in the context below
- PROCEED with the analysis using the information provided

Your role is to analyze extracted veteran data against VA rating criteria to identify:
1. Underrated conditions (current rating too low based on evidence)
2. Missed conditions (present in evidence but not rated)
3. Secondary conditions (caused by service-connected conditions)
4. Rating discrepancies and CFR inconsistencies

IMPORTANT MEDICAL-ONLY HANDLING:
- If medical conditions are present but CURRENT VA RATING DATA is missing, do NOT claim "no data".
- In that case, provide:
    1) Medical conditions found with evidence citations
    2) Missing VA evidence checklist (VA decision/code sheet, C&P/DBQ, service-connection/nexus evidence)
    3) Explicit statement that claim-ready rating analysis is limited until missing evidence is provided
- In medical-only mode, create one numbered finding block for each medically supported condition from VRIS-A section 2 and any checklist conditions. Do not collapse the packet into a short representative subset.
- Do not fabricate current VA ratings when they are not present in the documents.

VRIS System Documentation and CFR Rules (USE THIS):
{system_context}

Veteran Extracted Data and Evidence (ANALYZE THIS):
{veteran_context}

Analysis Request: {question}

You MUST provide your analysis with:
- Specific CFR § references (Title 38)
- Diagnostic Code citations
- Evidence-to-rating comparisons
- Confidence score (0-100%) for each finding
- Distinction between Phase 1 (strong, CFR-aligned) and Phase 2 (plausible but weaker)

Format each finding as:
1. Condition Name (Current Rating: X%)
   - Evidence: [list specific evidence from documents]
   - Confidence Score: X%
   - Phase: 1 or 2
   - Potential Rating: X%
   - CFR Citations: [list CFR sections]

If VA rating/service-connection evidence is insufficient, add a section:
MEDICAL-ONLY GAP SUMMARY:
- Conditions identified from records
- Missing claim-critical evidence
- Next required documents for full VRIS rating analysis

Be precise, evidence-based, and always cite CFR sections.
PROCEED WITH ANALYSIS NOW using the context provided above.
If the query contains checklist conditions, address each one explicitly and avoid "Evidence: Not specified" when page hints exist.
"""
        
        prompt = ChatPromptTemplate.from_template(template)
        
        def format_docs(docs):
            formatted = []
            for doc in docs:
                source = doc.metadata.get('filename', 'Unknown')
                page = doc.metadata.get('page', 'Unknown')
                snippet = truncate_for_prompt(doc.page_content, max_chars=700)
                formatted.append(f"[{source}, Page {page}]\n{snippet}")
            return "\n\n" + "-"*60 + "\n\n".join(formatted)
        
        self.vris_b_chain = (
            {
                "system_context": system_retriever | format_docs,
                "veteran_context": veteran_retriever | format_docs,
                "question": RunnablePassthrough()
            }
            | prompt
            | self.llm
            | StrOutputParser()
        )
        
        print("✓ VRIS-B Reasoning Pipeline initialized")
    
    def vris_a_extract(self, query: str) -> str:
        """Run VRIS-A extraction pipeline"""
        if not self.vris_a_chain:
            raise ValueError("VRIS-A not initialized. Run setup_vris_a_extraction_chain()")
        
        print("\n" + "="*70)
        print("VRIS-A EXTRACTION PIPELINE")
        print("="*70)
        print(f"Query: {query}\n")
        
        result = self.vris_a_chain.invoke(query)
        
        print("Extraction Complete")
        print("="*70 + "\n")
        
        return result
    
    def vris_b_analyze(self, query: str) -> str:
        """Run VRIS-B reasoning pipeline"""
        if not self.vris_b_chain:
            raise ValueError("VRIS-B not initialized. Run setup_vris_b_reasoning_chain()")
        
        print("\n" + "="*70)
        print("VRIS-B REASONING PIPELINE")
        print("="*70)
        print(f"Analysis: {query}\n")
        
        try:
            result = self.vris_b_chain.invoke(query)
        except Exception as e:
            error_text = str(e).lower()
            is_context_error = (
                "context_length_exceeded" in error_text
                or "maximum context length" in error_text
            )

            if not is_context_error:
                raise

            print("⚠️  VRIS-B context length exceeded. Retrying with compact reasoning mode...")

            if not self.system_vectorstore or not self.veteran_vectorstore:
                raise

            compact_system_retriever = self.system_vectorstore.as_retriever(
                search_type="similarity",
                search_kwargs={"k": 2},
            )
            compact_veteran_retriever = self.veteran_vectorstore.as_retriever(
                search_type="similarity",
                search_kwargs={"k": 4},
            )

            compact_template = """You are VRIS-B in compact fallback mode.

Analyze the provided veteran evidence against VA criteria with concise output.
If current VA ratings are missing, provide medical-only gap guidance and do not fabricate ratings.

System CFR context:
{system_context}

Veteran context:
{veteran_context}

Analysis request:
{question}

Return concise structured findings with evidence citations.
"""

            compact_prompt = ChatPromptTemplate.from_template(compact_template)

            def compact_format_docs(docs):
                lines = []
                for doc in docs:
                    source = doc.metadata.get('filename', 'Unknown')
                    page = doc.metadata.get('page', 'Unknown')
                    snippet = truncate_for_prompt(doc.page_content, max_chars=450)
                    lines.append(f"[{source}, Page {page}]\n{snippet}")
                return "\n\n".join(lines)

            compact_chain = (
                {
                    "system_context": compact_system_retriever | compact_format_docs,
                    "veteran_context": compact_veteran_retriever | compact_format_docs,
                    "question": RunnablePassthrough(),
                }
                | compact_prompt
                | self.llm
                | StrOutputParser()
            )

            compact_query = truncate_for_prompt(query, max_chars=2200)
            result = compact_chain.invoke(compact_query)
        
        print("Analysis Complete")
        print("="*70 + "\n")
        
        return result
    
    def dual_pipeline_analysis(self, extraction_query: str, reasoning_query: str) -> Dict[str, Any]:
        """
        Run both VRIS-A and VRIS-B, then compare results
        
        Returns agreement score and reconciled findings
        """
        print("\n" + "🔄 "*35)
        print("DUAL-PIPELINE VRIS ANALYSIS")
        print("🔄 "*35 + "\n")
        
        # Run VRIS-A
        vris_a_result = self.vris_a_extract(extraction_query)
        vris_a_result = self.append_condition_coverage_appendix(
            vris_a_result,
            "SUPPLEMENTAL CONDITION COVERAGE INDEX:",
            max_candidates=12,
        )

        vris_a_reasoning_payload = self.prepare_vris_a_for_reasoning(vris_a_result)
        
        # Run VRIS-B with VRIS-A's extraction included in the query
        vris_b_query = f"""
VRIS-A EXTRACTED DATA:
{vris_a_reasoning_payload}

---

{reasoning_query}
"""
        vris_b_result = self.vris_b_analyze(vris_b_query)
        if "current va rating data: not found" in vris_a_result.lower():
            vris_b_result = self.append_condition_coverage_appendix(
                vris_b_result,
                "SUPPLEMENTAL MEDICAL RECORD CONDITION INDEX:",
                max_candidates=12,
            )
        
        # In production, implement sophisticated agreement scoring
        # For now, return both results
        result = {
            "vris_a_extraction": vris_a_result,
            "vris_b_reasoning": vris_b_result,
            "agreement_score": None,  # Implement agreement calculation
            "reconciled_findings": None  # Implement reconciliation logic
        }
        
        print("\n✅ Dual-pipeline analysis complete")
        print("Note: Agreement scoring requires additional implementation\n")
        
        return result
    
    def generate_free_snapshot(self) -> Dict[str, Any]:
        """
        Generate Free Rating Snapshot (high-level summary)
        """
        extraction_query = """
        Extract from all veteran documents:
        1. Current combined VA disability rating
        2. List of all service-connected conditions with their individual ratings
        3. Any conditions mentioned in medical records that are NOT currently rated
        4. Key symptoms or limitations described
        """
        
        reasoning_query = """
        Based on the veteran's documents:
        1. Identify how many conditions appear potentially underrated
        2. Identify how many conditions are in evidence but not currently rated
        3. Identify any obvious secondary condition opportunities
        4. Estimate potential rating range (do not provide detailed CFR analysis)
        """
        
        result = self.dual_pipeline_analysis(extraction_query, reasoning_query)
        
        return {
            "snapshot_type": "Free Rating Snapshot",
            "extraction": result["vris_a_extraction"],
            "high_level_analysis": result["vris_b_reasoning"],
            "note": "This is a high-level summary. Full CFR analysis requires paid VRE."
        }
    
    def generate_initial_vre(self) -> Dict[str, Any]:
        """
        Generate Initial VRE (for first-time filers)
        """
        extraction_query = """
        From service treatment records and medical documentation, extract:
        1. ALL conditions with service connection evidence (in-service events, diagnoses)
        2. All symptoms, functional impairments, and limitations
        3. Temporal relationships (when conditions started, progression)
        4. All test results, ROM measurements, severity indicators
        5. Potential secondary conditions based on causal relationships
        """
        
        reasoning_query = """
        For a veteran filing their initial VA disability claim:
        1. Identify all service-connected conditions with supporting evidence
        2. Suggest likely initial rating bands for each condition based on CFR criteria
        3. Identify secondary conditions with strong medical support
        4. Calculate estimated combined rating
        5. Provide condition-by-condition breakdown with CFR references
        6. Include recommended claim language frameworks
        
        Phase 1: Strong, well-documented conditions
        Phase 2: Medically plausible but requiring VSO judgment
        """
        
        result = self.dual_pipeline_analysis(extraction_query, reasoning_query)
        
        return {
            "vre_type": "Initial VRE (First-Time Filer)",
            "extraction": result["vris_a_extraction"],
            "detailed_analysis": result["vris_b_reasoning"],
            "vris_a_result": result["vris_a_extraction"],
            "vris_b_result": result["vris_b_reasoning"]
        }
    
    def generate_second_look_vre(self) -> Dict[str, Any]:
        """
        Generate Second Look VRE (for already-rated veterans)
        """
        coverage_hint = self.build_condition_coverage_hint(max_candidates=12)

        extraction_query = f"""
        Extract from all provided documents (VA decision docs, C&P, DBQ, private medical records):
        1. Current VA rating profile:
            - If present, extract ALL service-connected conditions with current ratings, DCs, and effective dates
            - If absent, explicitly state: "Current VA rating data: NOT FOUND in provided documents"
        2. Medical evidence regardless of VA rating status:
            - Diagnosed conditions and diagnosis codes
            - Symptoms, severity, functional impact, labs/imaging/clinical findings
        3. Conditions in evidence that may be unrated or require additional claim evidence
        4. Potential secondary relationships
        5. Missing-evidence checklist for claim-ready VRIS analysis

        CRITICAL:
        - Do not output "no medical conditions" when diagnoses/assessments are present.
        - If VA rating data is missing, still provide full medical-condition extraction.
        - For each listed condition, include at least one evidence citation with source/page.
        - Never write "Evidence: Not specified" for a condition that appears in checklist pages.

        CONDITION COVERAGE CHECKLIST (MANDATORY TO ADDRESS):
        {coverage_hint}
        """

        reasoning_query = """
        Using the VRIS-A extraction above, perform one of two paths:

        PATH A - If current VA ratings are present:
        1. Identify conditions where evidence supports higher rating than currently assigned (UNDERRATED)
        2. Map current ratings to CFR criteria - identify underrating gaps
        3. Identify conditions present in evidence but with NO current rating (MISSED CONDITIONS)
        4. Identify secondary conditions with causal linkage to rated conditions
        5. Calculate potential increased combined rating
        6. Provide detailed CFR citations and evidence mappings
        7. Confidence scores for each finding

        PATH B - If current VA ratings are NOT present:
        1. List every medically supported condition found in records as its own numbered finding block with evidence citations
        2. Provide "MEDICAL-ONLY GAP SUMMARY" with missing evidence:
            - VA Decision Letter/Code Sheet
            - C&P exam or DBQ for key conditions
            - Service-connection evidence (in-service event + nexus)
        3. State clearly that claim-ready rating opportunity scoring is limited until missing VA evidence is provided
        4. Use the format "Condition Name (Not Rated)" for each medical-only finding block
        5. Do not fabricate current ratings or label all data as absent

        IMPORTANT:
        - If a condition has a current rating (even 0%), it goes in UNDERRATED category
        - Only conditions with NO rating at all go in MISSED CONDITIONS
        - Always check VRIS-A section 1 for the current rating before labeling as "Not Rated"
        - For medical-only packets, use VRIS-A extracted conditions and citations as your source of truth.

        Phase 1: Strong increase opportunities (90%+ confidence)
        Phase 2: Plausible increases requiring VSO evaluation
        """
        
        result = self.dual_pipeline_analysis(extraction_query, reasoning_query)
        
        return {
            "vre_type": "Second Look VRE (Already-Rated Veteran)",
            "extraction": result["vris_a_extraction"],
            "detailed_analysis": result["vris_b_reasoning"],
            "vris_a_result": result["vris_a_extraction"],
            "vris_b_result": result["vris_b_reasoning"]
        }
    
    def initialize(self, force_reload_system: bool = False):
        """
        Initialize VRIS RAG system
        
        Args:
            force_reload_system: Force rebuilding system vectorstore (normally False)
        
        Note:
            - System documents are pre-built and persisted
            - Veteran documents are processed on-demand (not persisted)
        """
        print("\n" + "🚀 "*35)
        print("INITIALIZING VRIS™ RAG SYSTEM")
        print("Veteran Rating Intelligence System")
        print("🚀 "*35 + "\n")
        
        # Load or build system vectorstore (persisted)
        self.create_system_vectorstore(force_reload=force_reload_system)
        
        # Process veteran documents (NOT persisted, fresh each time)
        self.create_veteran_vectorstore(force_reload=True)
        
        # Setup dual pipelines
        if self.veteran_vectorstore:
            self.setup_vris_a_extraction_chain()
        else:
            print("⚠️  Warning: No veteran documents loaded. VRIS-A will not be available.")
        
        if self.system_vectorstore and self.veteran_vectorstore:
            self.setup_vris_b_reasoning_chain()
        elif not self.veteran_vectorstore:
            print("⚠️  Warning: VRIS-B requires veteran documents. Please add documents to analyze.")
        
        print("\n" + "✅ "*35)
        if self.veteran_vectorstore:
            print("VRIS™ RAG SYSTEM READY")
        else:
            print("VRIS™ RAG SYSTEM READY (Limited - Add veteran documents for full functionality)")
        print("✅ "*35 + "\n")


def main():
    """Demonstrate VRIS RAG system"""
    
    # Configuration
    SYSTEM_DOCS_FOLDER = "./system-doc"  # VRIS documentation, CFR rules
    VETERAN_DOCS_FOLDER = "./user_pdfs/1"  # Veteran uploaded documents
    CHROMA_DB_DIR = "./chroma_db"
    
    # Ensure folders exist
    os.makedirs(SYSTEM_DOCS_FOLDER, exist_ok=True)
    os.makedirs(VETERAN_DOCS_FOLDER, exist_ok=True)
    
    # Initialize VRIS RAG system
    vris = VRISRAGSystem(
        system_docs_folder=SYSTEM_DOCS_FOLDER,
        veteran_docs_folder=VETERAN_DOCS_FOLDER,
        persist_directory=CHROMA_DB_DIR,
        model_name="gpt-4"  # Use GPT-4 for production
    )
    
    # Initialize (set force_reload_system=True to reprocess all documents)
    vris.initialize(force_reload_system=False)
    
    # Example workflows
    print("\n" + "="*70)
    print("VRIS EVALUATION WORKFLOWS")
    print("="*70)
    print("\nAvailable workflows:")
    print("  1. Free Rating Snapshot")
    print("  2. Initial VRE (First-Time Filer)")
    print("  3. Second Look VRE (Already-Rated)")
    print("  4. Custom VRIS-A Extraction")
    print("  5. Custom VRIS-B Analysis")
    print("  6. Interactive Query Mode")
    print("  7. Exit")
    
    while True:
        print("\n" + "-"*70)
        choice = input("\nSelect workflow (1-7): ").strip()
        
        if choice == '1':
            print("\n🔍 Generating Free Rating Snapshot...")
            snapshot = vris.generate_free_snapshot()
            print("\n" + "="*70)
            print("FREE RATING SNAPSHOT RESULTS")
            print("="*70)
            print(f"\n{snapshot['extraction']}")
            print(f"\n{snapshot['high_level_analysis']}")
            print(f"\n{snapshot['note']}")
        
        elif choice == '2':
            print("\n📋 Generating Initial VRE (First-Time Filer)...")
            initial_vre = vris.generate_initial_vre()
            print("\n" + "="*70)
            print("INITIAL VRE RESULTS")
            print("="*70)
            print("\nVRIS-A EXTRACTION:")
            print(initial_vre['vris_a_result'])
            print("\n\nVRIS-B REASONING:")
            print(initial_vre['vris_b_result'])
        
        elif choice == '3':
            print("\n🔎 Generating Second Look VRE (Already-Rated)...")
            second_look = vris.generate_second_look_vre()
            print("\n" + "="*70)
            print("SECOND LOOK VRE RESULTS")
            print("="*70)
            print("\nVRIS-A EXTRACTION:")
            print(second_look['vris_a_result'])
            print("\n\nVRIS-B REASONING:")
            print(second_look['vris_b_result'])
        
        elif choice == '4':
            query = input("\nEnter VRIS-A extraction query: ").strip()
            if query:
                result = vris.vris_a_extract(query)
                print(f"\n{result}")
        
        elif choice == '5':
            query = input("\nEnter VRIS-B reasoning query: ").strip()
            if query:
                result = vris.vris_b_analyze(query)
                print(f"\n{result}")
        
        elif choice == '6':
            print("\n💬 Interactive Query Mode")
            print("Type 'back' to return to main menu\n")
            while True:
                query = input("\nQuery: ").strip()
                if query.lower() == 'back':
                    break
                if query:
                    # Default to VRIS-B reasoning for general queries
                    result = vris.vris_b_analyze(query)
                    print(f"\n{result}")
        
        elif choice == '7':
            print("\n👋 VRIS™ session ended")
            break
        
        else:
            print("Invalid choice. Please select 1-7.")


if __name__ == "__main__":
    main()
