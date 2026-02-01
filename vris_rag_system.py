"""
VRIS™ (Veteran Rating Intelligence System) RAG Implementation
Dual-pipeline AI system for VA disability rating analysis
"""

import os
from pathlib import Path
from typing import List, Dict, Any, Optional
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader, Docx2txtLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_chroma import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
import json

# Load environment variables
load_dotenv()


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
        
        # Store classified documents
        self.classified_docs = {}
    
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
        
        # Load PDF files
        for pdf_file in pdf_files:
            print(f"Loading PDF: {pdf_file.name}")
            try:
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
        for file_path in document_paths:
            path = Path(file_path)
            if not path.exists():
                print(f"⚠️  File not found: {file_path}")
                continue
            
            print(f"Loading: {path.name}")
            try:
                if path.suffix.lower() == '.pdf':
                    loader = PyPDFLoader(str(path))
                elif path.suffix.lower() == '.docx':
                    loader = Docx2txtLoader(str(path))
                elif path.suffix.lower() == '.txt':
                    with open(path, 'r', encoding='utf-8') as f:
                        content = f.read()
                    from langchain_core.documents import Document
                    doc = Document(
                        page_content=content,
                        metadata={'filename': path.name, 'source': str(path), 'file_type': 'txt'}
                    )
                    documents.append(doc)
                    continue
                else:
                    print(f"⚠️  Unsupported file type: {path.suffix}")
                    continue
                
                docs = loader.load()
                for doc in docs:
                    doc.metadata['filename'] = path.name
                    doc.metadata['source'] = str(path)
                documents.extend(docs)
                
            except Exception as e:
                print(f"❌ Error loading {path.name}: {e}")
        
        if not documents:
            raise ValueError("No documents could be loaded from the provided paths")
        
        print(f"Loaded {len(documents)} document(s)")
        
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
            search_kwargs={"k": 10}  # More context for complete extraction
        )
        
        template = """You are VRIS-A, the extraction component of the Veteran Rating Intelligence System.

Your role is to extract ALL rating-relevant data from veteran documents with precision and completeness.

IMPORTANT: You are being provided with actual veteran document content below. Extract ALL information present.

Extract the following information from the provided context:

1. CURRENT VA RATING DATA:
   - Combined disability rating percentage
   - Individual condition ratings and percentages (CHECK FOR: PTSD, back/spine, knees, sleep conditions, etc.)
   - Diagnostic Codes (DC) for each condition
   - Effective dates
   - Service-connected conditions list

2. MEDICAL CONDITIONS:
   - All diagnosed conditions (current and historical) - INCLUDING mental health and sleep conditions
   - Symptoms and severity indicators
   - Functional impairments and limitations
   - ROM (Range of Motion) measurements if applicable
   - Test results (labs, imaging, sleep studies, etc.)

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

Context from veteran documents:
{context}

Query: {question}

Provide a structured extraction in clear, organized format. Be thorough and precise.
Extract ALL conditions mentioned across all documents - do not omit mental health, sleep disorders, or any other diagnosed conditions.
Extract only what is explicitly stated - do not infer or reason about ratings.
"""
        
        prompt = ChatPromptTemplate.from_template(template)
        
        def format_docs(docs):
            formatted = []
            for doc in docs:
                source = doc.metadata.get('filename', 'Unknown')
                page = doc.metadata.get('page', 'Unknown')
                doc_type = doc.metadata.get('document_type', 'Unknown')
                formatted.append(f"[{doc_type} - {source}, Page {page}]\n{doc.page_content}")
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
            search_kwargs={"k": 6}
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

Be precise, evidence-based, and always cite CFR sections.
PROCEED WITH ANALYSIS NOW using the context provided above.
"""
        
        prompt = ChatPromptTemplate.from_template(template)
        
        def format_docs(docs):
            formatted = []
            for doc in docs:
                source = doc.metadata.get('filename', 'Unknown')
                page = doc.metadata.get('page', 'Unknown')
                formatted.append(f"[{source}, Page {page}]\n{doc.page_content}")
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
        
        result = self.vris_b_chain.invoke(query)
        
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
        
        # Run VRIS-B
        vris_b_result = self.vris_b_analyze(reasoning_query)
        
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
        extraction_query = """
        Extract from VA Decision Letter and all supporting documents:
        1. Current VA rating profile (all conditions, DCs, percentages, effective dates)
        2. All evidence from medical records, C&P exams, DBQs
        3. Any conditions in evidence that are NOT currently rated
        4. Severity indicators that may exceed current rating criteria
        5. Potential secondary conditions
        """
        
        reasoning_query = """
        Compare veteran's current VA rating to evidence:
        1. Identify conditions where evidence supports higher rating than assigned
        2. Map current ratings to CFR criteria - identify underrating
        3. Identify conditions present in evidence but not rated (missed conditions)
        4. Identify secondary conditions with causal linkage to rated conditions
        5. Calculate potential increased combined rating
        6. Provide detailed CFR citations and evidence mappings
        7. Confidence scores for each finding
        
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
    
    # Initialize (set force_reload=True to reprocess all documents)
    vris.initialize(force_reload=False)
    
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
