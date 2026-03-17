"""
VRIS™ Simple API - Single Endpoint
Upload veteran documents → Get complete analysis
"""

from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from typing import List, Dict, Any
from pathlib import Path
import shutil
import os
import re
from datetime import datetime
import tempfile
from vris_rag_system import VRISRAGSystem

app = FastAPI(
    title="VRIS™ API", 
    version="1.0.0",
    description="Veteran Rating Intelligence System - AI-driven VA disability rating analysis"
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Configuration
SYSTEM_DOCS_FOLDER = "./system-doc"
CHROMA_DB_DIR = "./chroma_db"

# Pre-load system vectorstore at startup (only once)
print("\n" + "="*70)
print("🚀 VRIS™ API Server Starting...")
print("="*70)
print("Loading VRIS system documentation and CFR rules...")

vris_system = VRISRAGSystem(
    system_docs_folder=SYSTEM_DOCS_FOLDER,
    veteran_docs_folder="",
    persist_directory=CHROMA_DB_DIR,
    model_name="gpt-4"
)
vris_system.create_system_vectorstore(force_reload=False)

print("✓ System vectorstore ready")
print("✓ VRIS ready to analyze veteran documents")
print("="*70 + "\n")


def _normalize_condition_key(name: str) -> str:
    normalized = re.sub(r"[^a-z0-9]+", " ", (name or "").lower()).strip()
    return re.sub(r"\s+", " ", normalized)


def _append_parsed_condition(findings: Dict[str, Any], condition: Dict[str, Any] | None) -> None:
    if not condition or not condition.get("name"):
        return

    if condition.get("current_rating") == "Not Rated":
        findings["missed_conditions"].append(condition)
    elif "secondary" in condition["name"].lower():
        findings["secondary_conditions"].append(condition)
    elif condition.get("current_rating") and condition.get("potential_rating"):
        findings["underrated_conditions"].append(condition)


def _merge_unique_condition_entries(entries: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    merged: Dict[tuple[str, str, str], Dict[str, Any]] = {}
    order: List[tuple[str, str, str]] = []

    for entry in entries:
        name = (entry or {}).get("name", "").strip()
        if not name:
            continue

        current_rating = (entry.get("current_rating") or "").strip()
        potential_rating = (entry.get("potential_rating") or "").strip()
        key = (
            _normalize_condition_key(name),
            current_rating.lower(),
            potential_rating.lower(),
        )

        if key not in merged:
            merged[key] = {
                "name": name,
                "current_rating": entry.get("current_rating"),
                "potential_rating": entry.get("potential_rating"),
                "confidence": entry.get("confidence"),
                "phase": entry.get("phase"),
                "evidence": list(dict.fromkeys(entry.get("evidence", []))),
                "cfr_citations": list(dict.fromkeys(entry.get("cfr_citations", []))),
            }
            order.append(key)
            continue

        existing = merged[key]
        if existing.get("confidence") is None or (
            entry.get("confidence") is not None and entry["confidence"] > existing["confidence"]
        ):
            existing["confidence"] = entry.get("confidence")

        if not existing.get("phase") and entry.get("phase") is not None:
            existing["phase"] = entry.get("phase")

        if not existing.get("current_rating") and entry.get("current_rating"):
            existing["current_rating"] = entry.get("current_rating")

        if not existing.get("potential_rating") and entry.get("potential_rating"):
            existing["potential_rating"] = entry.get("potential_rating")

        existing["evidence"] = list(dict.fromkeys(existing["evidence"] + entry.get("evidence", [])))
        existing["cfr_citations"] = list(dict.fromkeys(existing["cfr_citations"] + entry.get("cfr_citations", [])))

    return [merged[key] for key in order]


def _dedupe_findings(findings: Dict[str, Any]) -> Dict[str, Any]:
    findings["underrated_conditions"] = _merge_unique_condition_entries(findings.get("underrated_conditions", []))
    findings["missed_conditions"] = _merge_unique_condition_entries(findings.get("missed_conditions", []))
    findings["secondary_conditions"] = _merge_unique_condition_entries(findings.get("secondary_conditions", []))
    findings["total_opportunities"] = (
        len(findings["underrated_conditions"]) +
        len(findings["missed_conditions"]) +
        len(findings["secondary_conditions"])
    )
    return findings


def _parse_condition_coverage_index(text: str) -> List[Dict[str, Any]]:
    titles = {
        "SUPPLEMENTAL CONDITION COVERAGE INDEX:",
        "SUPPLEMENTAL MEDICAL RECORD CONDITION INDEX:",
    }
    item_pattern = re.compile(
        r"^\d+\.\s+(?P<name>.+?)\s+\|\s+ICD:\s+(?P<codes>.+?)\s+\|\s+Pages:\s+(?P<pages>.+?)$"
    )

    conditions: List[Dict[str, Any]] = []
    in_section = False

    for raw_line in (text or "").splitlines():
        line = raw_line.strip()

        if line in titles:
            in_section = True
            continue

        if not in_section:
            continue

        if not line or line.lower().startswith("checklist generated from deterministic"):
            continue

        match = item_pattern.match(line)
        if not match:
            if line.endswith(":") and not re.match(r"^\d+\.", line):
                in_section = False
            continue

        name = match.group("name").strip(" -:;,.\t()[]")
        codes = [code.strip() for code in match.group("codes").split(",") if code.strip() and code.strip().lower() != "none"]
        pages = [int(page.strip()) for page in match.group("pages").split(",") if page.strip().isdigit()]
        codes_text = ", ".join(codes) if codes else "ICD not listed"
        pages_text = ", ".join(str(page) for page in pages) if pages else "Unknown"
        page_label = "Page" if len(pages) == 1 else "Pages"

        conditions.append(
            {
                "name": name,
                "current_rating": "Not Rated",
                "potential_rating": "Not specified due to lack of VA rating/service-connection evidence",
                "confidence": None,
                "phase": None,
                "evidence": [f"{name} - {codes_text} ({page_label} {pages_text})"],
                "cfr_citations": [],
            }
        )

    return _merge_unique_condition_entries(conditions)


def _is_medical_only_packet(vris_a_output: str, vris_b_output: str) -> bool:
    combined = f"{vris_a_output}\n{vris_b_output}".lower()
    return (
        "current va rating data: not found" in combined and
        "medical-only gap summary" in combined
    )


def _enrich_medical_only_findings(findings: Dict[str, Any], vris_a_output: str, vris_b_output: str) -> Dict[str, Any]:
    findings["missed_conditions"].extend(_parse_condition_coverage_index(vris_a_output))
    findings["missed_conditions"].extend(_parse_condition_coverage_index(vris_b_output))
    return _dedupe_findings(findings)


@app.post("/api/vris/analyze")
async def analyze_veteran_documents(
    files: List[UploadFile] = File(
        ..., 
        description="Veteran documents (VA Decision Letter, C&P Exams, DBQs, Medical Records)"
    )
) -> Dict[str, Any]:
    """
    VRIS™ Complete Analysis Endpoint
    
    Upload veteran documents and receive comprehensive analysis identifying:
    - Underrated conditions (current rating too low)
    - Missed conditions (in evidence but not rated)
    - Secondary conditions (caused by service-connected conditions)
    
    VRIS-A extracts all data (diagnostic codes, CFR references, symptoms)
    VRIS-B reasons through findings and flags inconsistencies
    
    Both pipelines must agree by 90%+ for high-confidence opportunities.
    """
    temp_dir = None
    
    try:
        # Validate files
        if not files or len(files) == 0:
            raise HTTPException(
                status_code=400, 
                detail="No files provided. Please upload veteran documents."
            )
        
        # Create temporary directory for this analysis
        temp_dir = tempfile.mkdtemp(prefix="vris_")
        uploaded_files = []
        file_info = []
        
        # Save uploaded files
        for file in files:
            file_ext = Path(file.filename).suffix.lower()
            if file_ext not in ['.pdf', '.docx', '.txt']:
                print(f"⚠️  Skipping unsupported file type: {file.filename}")
                continue
            
            file_path = Path(temp_dir) / file.filename
            with open(file_path, "wb") as buffer:
                shutil.copyfileobj(file.file, buffer)
            
            uploaded_files.append(str(file_path))
            file_info.append({
                "filename": file.filename,
                "size": os.path.getsize(file_path),
                "type": file_ext[1:]
            })
        
        if not uploaded_files:
            raise HTTPException(
                status_code=400, 
                detail="No valid files uploaded. Supported formats: PDF, DOCX, TXT"
            )
        
        print(f"\n{'='*70}")
        print(f"📄 Analyzing {len(uploaded_files)} document(s) for veteran")
        print(f"{'='*70}")
        
        # Create VRIS instance for this analysis
        vris = VRISRAGSystem(
            system_docs_folder=SYSTEM_DOCS_FOLDER,
            veteran_docs_folder="",
            persist_directory=CHROMA_DB_DIR,
            model_name="gpt-4"
        )
        
        # Use pre-loaded system vectorstore
        vris.system_vectorstore = vris_system.system_vectorstore
        
        # Process veteran documents (in-memory, not persisted)
        vris.process_veteran_documents_from_upload(uploaded_files)
        
        print("\n🔄 Running VRIS dual-pipeline analysis...")
        print("   VRIS-A: Extracting diagnostic codes, symptoms, evidence...")
        
        # Run complete VRIS analysis (Second Look VRE for comprehensive analysis)
        analysis_result = vris.generate_second_look_vre()
        
        print("   VRIS-B: Reasoning against CFR Title 38 criteria...")
        
        # Extract findings and structure response
        vris_a_data = analysis_result.get('vris_a_result', '')
        vris_b_data = analysis_result.get('vris_b_result', '')
        
        # Check if VRIS-B failed to analyze
        if "I don't have the ability" in vris_b_data or "I can't" in vris_b_data or len(vris_b_data) < 200:
            print("⚠️  Warning: VRIS-B analysis may have failed - retrying with explicit instructions...")
            # This shouldn't happen with the updated prompts, but log it
            raise Exception("VRIS-B refused to analyze. Please check system vectorstore has CFR documentation.")
        
        # Parse VRIS-B output to extract structured findings
        findings = parse_vris_findings(vris_b_data)

        medical_only = _is_medical_only_packet(vris_a_data, vris_b_data)
        if medical_only:
            findings = _enrich_medical_only_findings(findings, vris_a_data, vris_b_data)

        # Generate summary with medical-only fallback guidance
        summary = generate_summary(findings)
        combined_analysis_text = f"{vris_a_data}\n{vris_b_data}".lower()
        medical_only_signals = [
            "current va rating data: not found",
            "medical-only gap summary",
            "missing claim-critical evidence",
            "insufficient va rating",
            "service-connection evidence",
        ]
        if medical_only or any(signal in combined_analysis_text for signal in medical_only_signals):
            summary["analysis_mode"] = "medical-evidence-only"
            summary["recommendation"] = (
                "Medical conditions found, but VA rating/service-connection evidence is missing. "
                "Upload VA Decision Letter/Code Sheet, relevant C&P or DBQ documents, "
                "and service-connection evidence (in-service event + nexus)."
            )
        
        # Build response
        response = {
            "success": True,
            "timestamp": datetime.now().isoformat(),
            "analysis_type": "VRIS Complete Analysis",
            "files_analyzed": file_info,
            
            # VRIS-A: Raw extraction
            "vris_a_extraction": {
                "description": "Extracted data from veteran documents",
                "data": vris_a_data
            },
            
            # VRIS-B: Reasoning and findings
            "vris_b_reasoning": {
                "description": "Analysis against VA rating criteria (38 CFR)",
                "data": vris_b_data
            },
            
            # Structured findings
            "findings": findings,
            
            # Summary
            "summary": summary,
            
            # Compliance
            "data_retention": "Files processed in-memory only. Not stored. Deleted immediately after analysis.",
            "system_info": "VRIS™ - Veteran Rating Intelligence System"
        }
        
        print("✅ Analysis complete")
        print(f"{'='*70}\n")
        
        return response
    
    except HTTPException:
        raise
    except Exception as e:
        error_text = str(e)
        print(f"❌ Error during analysis: {error_text}")

        # OCR/setup issues should be user-actionable (400), not server-fault (500)
        if "OCR is required" in error_text or "Install OCR dependencies" in error_text:
            raise HTTPException(status_code=400, detail=error_text)

        # Token-limit errors should be surfaced as actionable request issues.
        lowered = error_text.lower()
        if "context_length_exceeded" in lowered or "maximum context length" in lowered:
            raise HTTPException(
                status_code=400,
                detail=(
                    "Analysis input exceeded model context limit. "
                    "Retry with fewer/lower-length documents or use a higher-context model configuration."
                ),
            )

        raise HTTPException(status_code=500, detail=f"Analysis failed: {error_text}")
    
    finally:
        # Cleanup: Delete temporary files immediately
        if temp_dir and os.path.exists(temp_dir):
            try:
                shutil.rmtree(temp_dir)
                print(f"🗑️  Temporary files deleted: {temp_dir}")
            except Exception as e:
                print(f"⚠️  Warning: Could not delete temp files: {e}")


def parse_vris_findings(vris_b_output: str) -> Dict[str, Any]:
    """
    Parse VRIS-B output to extract structured findings
    """
    findings = {
        "underrated_conditions": [],
        "missed_conditions": [],
        "secondary_conditions": [],
        "total_opportunities": 0
    }
    
    lines = vris_b_output.split('\n')
    current_condition = None
    condition_header_pattern = re.compile(
        r"^\d+\.\s+(?P<name>.+?)\s+\((?:Current Rating:\s*(?P<current_rating>[^)]+)|(?P<not_rated>Not Rated))\)\s*$"
    )
    
    for line in lines:
        line = line.strip()

        if line in {"SUPPLEMENTAL MEDICAL RECORD CONDITION INDEX:", "SUPPLEMENTAL CONDITION COVERAGE INDEX:"}:
            break
        
        # Look for condition headers (numbered items)
        header_match = condition_header_pattern.match(line)
        if header_match:
            _append_parsed_condition(findings, current_condition)
            current_condition = {
                "name": header_match.group("name").strip(),
                "current_rating": header_match.group("current_rating").strip() if header_match.group("current_rating") else "Not Rated",
                "potential_rating": None,
                "confidence": None,
                "phase": None,
                "evidence": [],
                "cfr_citations": []
            }
            continue
        
        # Extract potential rating
        elif current_condition and 'Potential Rating:' in line:
            try:
                potential = line.split('Potential Rating:')[1].strip().rstrip('%')
                current_condition["potential_rating"] = potential
            except:
                pass
        
        # Extract confidence score
        elif current_condition and 'Confidence Score:' in line:
            try:
                confidence = line.split('Confidence Score:')[1].strip().rstrip('%')
                current_condition["confidence"] = int(confidence)
            except:
                pass
        
        # Extract phase
        elif current_condition and 'Phase:' in line:
            try:
                phase = line.split('Phase:')[1].strip()
                current_condition["phase"] = int(phase) if phase.isdigit() else phase
            except:
                pass
        
        # Extract CFR citations
        elif current_condition and ('38 CFR' in line or 'Diagnostic Code' in line) and 'Evidence:' not in line:
            current_condition["cfr_citations"].append(line)
        
        # Extract evidence (make sure "Evidence:" lines go to evidence, not cfr_citations)
        elif current_condition and 'Evidence:' in line:
            evidence_text = line.split('Evidence:')[1].strip() if ':' in line else line
            if evidence_text:
                current_condition["evidence"].append(evidence_text)
        
        # Store condition when we hit a new one or end
        elif current_condition and (line.startswith('---') or line == ''):
            pass  # Don't store here, we'll store when we hit the next condition
    
    # Don't forget the last condition!
    _append_parsed_condition(findings, current_condition)
    return _dedupe_findings(findings)


def generate_summary(findings: Dict[str, Any]) -> Dict[str, Any]:
    """
    Generate executive summary of findings
    """
    high_confidence_count = sum(
        1 for cond in findings["underrated_conditions"] + findings["missed_conditions"] 
        if cond.get("confidence", 0) >= 90
    )
    
    return {
        "total_opportunities_identified": findings["total_opportunities"],
        "underrated_conditions_count": len(findings["underrated_conditions"]),
        "missed_conditions_count": len(findings["missed_conditions"]),
        "secondary_conditions_count": len(findings["secondary_conditions"]),
        "high_confidence_opportunities": high_confidence_count,
        "recommendation": (
            "Strong increase opportunities identified" 
            if high_confidence_count > 0 
            else "Review findings with VSO for potential claims"
        )
    }


@app.get("/api/vris/health")
async def health_check() -> Dict[str, Any]:
    """Health check endpoint"""
    return {
        "status": "healthy",
        "service": "VRIS™ - Veteran Rating Intelligence System",
        "system_ready": vris_system.system_vectorstore is not None,
        "version": "1.0.0",
        "timestamp": datetime.now().isoformat()
    }


@app.get("/")
async def root():
    """Root endpoint with API information"""
    return {
        "service": "VRIS™ API",
        "description": "Veteran Rating Intelligence System - AI-driven VA disability rating analysis",
        "version": "1.0.0",
        "endpoints": {
            "analyze": "POST /api/vris/analyze - Upload documents and get complete analysis",
            "health": "GET /api/vris/health - Health check",
            "docs": "GET /docs - Interactive API documentation"
        },
        "usage": {
            "1": "Upload veteran documents (VA Decision Letter, C&P Exams, DBQs, Medical Records)",
            "2": "VRIS analyzes documents using dual-pipeline (VRIS-A extraction + VRIS-B reasoning)",
            "3": "Receive structured analysis with underratings, missed conditions, and secondary opportunities",
            "4": "All findings backed by CFR Title 38 citations"
        },
        "data_privacy": "Documents processed in-memory only. Not stored. Deleted immediately after analysis."
    }


if __name__ == "__main__":
    import uvicorn
    
    print("\n" + "🎯"*35)
    print("VRIS™ Simple API - Single Endpoint Mode")
    print("Upload documents → Get complete analysis")
    print("🎯"*35 + "\n")
    
    uvicorn.run(app, host="0.0.0.0", port=8000, log_level="info")
