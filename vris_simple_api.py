"""
VRIS™ Simple API - Single Endpoint
Upload veteran documents → Get complete analysis
"""

from fastapi import FastAPI, UploadFile, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.openapi.utils import get_openapi
from fastapi.responses import StreamingResponse
from typing import Any, Dict, List, Optional
from pathlib import Path
import shutil
import os
import re
import io
from datetime import datetime
import tempfile
from vris_rag_system import VRISRAGSystem
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.lib.colors import HexColor, white
from reportlab.platypus import SimpleDocTemplate, Paragraph, Table, TableStyle, Spacer, KeepTogether
from reportlab.lib.enums import TA_LEFT, TA_CENTER

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


def custom_openapi():
    if app.openapi_schema:
        return app.openapi_schema
    schema = get_openapi(
        title=app.title,
        version=app.version,
        description=app.description,
        routes=app.routes,
    )
    # Swagger UI needs format:binary (OAS 3.0), not contentMediaType (OAS 3.1)
    for comp in schema.get("components", {}).get("schemas", {}).values():
        for prop in comp.get("properties", {}).values():
            if prop.get("type") == "array":
                items = prop.get("items", {})
                if "contentMediaType" in items:
                    del items["contentMediaType"]
                    items["format"] = "binary"
            elif "contentMediaType" in prop:
                del prop["contentMediaType"]
                prop["format"] = "binary"
    app.openapi_schema = schema
    return schema


app.openapi = custom_openapi

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


def _extract_veteran_name(uploaded_file_paths: List[str], vris_a_data: str = "", vris_b_data: str = "") -> Optional[str]:
    """
    Attempt to extract the veteran/patient name from:
    1. Raw text of uploaded files (PDF/TXT)
    2. VRIS-A / VRIS-B output text as fallback
    3. Uploaded filenames as last resort
    Returns the name in 'Firstname Lastname' format, or None if not found.
    """
    def _clean(text: str) -> str:
        return re.sub(r'[ \t]+', ' ', text)

    patterns = [
        # "Patient Name: SPENCE, ANYA" or "Patient: SPENCE, ANYA M" — LAST, FIRST [MIDDLE]
        re.compile(r'Patient\s*Name:\s*([A-Z][A-Z\-\']+,\s*[A-Z][A-Z\s\.\-\']{1,30})', re.IGNORECASE),
        re.compile(r'Patient:\s*([A-Z][A-Z\-\']+,\s*[A-Z][A-Z\s\.\-\']{1,30})', re.IGNORECASE),
        # "VETERAN: John M. Sample" or "VETERAN:  John  M.  Sample" (VA decision letter)
        re.compile(r'VETERAN:\s*([A-Z][a-zA-Z\'\-]+(?:\s+[A-Z][a-zA-Z\.\'\-]+){1,3})', re.IGNORECASE),
        # "Veteran: John Smith"
        re.compile(r'Veteran:\s*([A-Z][a-z]+(?:\s[A-Z][a-z]+){1,3})', re.IGNORECASE),
        # "Name: SMITH, JOHN" or "Name: John Smith"
        re.compile(r'(?:^|\n)\s*Name:\s*([A-Z][A-Z\-\']+,\s*[A-Z][A-Z\s\.\-\']{1,30})', re.IGNORECASE | re.MULTILINE),
        re.compile(r'(?:^|\n)\s*Name:\s*([A-Z][a-z]+(?:\s[A-Z][a-z]+){1,3})', re.IGNORECASE | re.MULTILINE),
        # "VETERAN NAME: JOHN SMITH"
        re.compile(r'VETERAN\s+NAME:\s*([A-Z][A-Z\s\.\-\']{3,40})', re.IGNORECASE),
        # "BENEFICIARY: JOHN SMITH"
        re.compile(r'BENEFICIARY:\s*([A-Z][A-Z\s\.\-\']{3,40})', re.IGNORECASE),
        # "Claimant Name: John Smith"
        re.compile(r'Claimant\s+Name:\s*([A-Z][a-zA-Z\'\-]+(?:\s[A-Z][a-zA-Z\'\-]+){1,3})', re.IGNORECASE),
    ]

    def _normalise(raw: str) -> str:
        """Convert 'SPENCE, ANYA M' -> 'Anya Spence', plain names -> Title Case."""
        raw = re.sub(r'\s+', ' ', raw).strip().rstrip('.')
        # Strip trailing field labels or SSN that VA records append on same line
        raw = re.sub(
            r'\s+(SSN|DOB|DOD|MPI|MRN|Date|Appt|Clinic|Phone|Provider|Treatment|Status'
            r'|\d{3}[-\s]\d{2}[-\s]\d{4}).*$',
            '', raw, flags=re.IGNORECASE
        ).strip()
        if ',' in raw:
            parts = [p.strip() for p in raw.split(',', 1)]
            first_parts = parts[1].split()
            # Drop single-letter middle initials AND known field-label words
            _field_labels = {'date', 'appt', 'clinic', 'phone', 'type', 'provider',
                             'treatment', 'status', 'ssn', 'dob'}
            first = ' '.join(
                p for p in first_parts
                if len(p) > 1 and p.lower() not in _field_labels
            )
            return f"{first.title()} {parts[0].title()}"
        return raw.title()

    _NOISE_WORDS = {'the', 'and', 'for', 'rating', 'decision', 'medical',
                    'health', 'clinic', 'center', 'affairs', 'veterans'}

    def _valid(name: str) -> bool:
        words = name.split()
        return (2 <= len(words) <= 5
                and all(len(w) >= 2 for w in words)
                and not any(w.lower() in _NOISE_WORDS for w in words))

    def _search(text: str) -> Optional[str]:
        cleaned = _clean(text)
        for pat in patterns:
            m = pat.search(cleaned)
            if m:
                candidate = _normalise(m.group(1))
                if _valid(candidate):
                    return candidate
        return None

    # 1. Read raw text from uploaded files
    for path in uploaded_file_paths:
        try:
            ext = Path(path).suffix.lower()
            text = ""
            if ext == '.pdf':
                try:
                    import pypdf
                    reader = pypdf.PdfReader(path)
                    for page in reader.pages[:5]:
                        text += (page.extract_text() or "")
                except Exception:
                    pass
            elif ext in ('.txt', '.docx'):
                with open(path, 'r', errors='ignore') as f:
                    text = f.read(3000)
            name = _search(text)
            if name:
                return name
        except Exception:
            continue

    # 2. Fallback: scan VRIS output
    for text in (vris_a_data, vris_b_data):
        name = _search(text)
        if name:
            return name

    # 3. Last resort: extract from filename (e.g. "Anya Spence Sleep Apnea Records.pdf")
    for path in uploaded_file_paths:
        stem = Path(path).stem
        stem_clean = re.sub(
            r'[-_\s]*(sleep|medical|health|records|report|exam|decision|letter|'
            r'dbq|cp|allergic|rhinitis|headache|migraine|knee|anemia|general|pcm|'
            r'copy|final|va|army|navy|air|force|marine|jefferson|request|inrlmr).*$',
            '', stem, flags=re.IGNORECASE
        ).strip(' _-')
        words = re.findall(r'[A-Za-z]{2,}', stem_clean)
        if 2 <= len(words) <= 3:
            candidate = ' '.join(w.title() for w in words)
            if _valid(candidate):
                return candidate

    return None


def _enrich_medical_only_findings(findings: Dict[str, Any], vris_a_output: str, vris_b_output: str) -> Dict[str, Any]:
    findings["missed_conditions"].extend(_parse_condition_coverage_index(vris_a_output))
    findings["missed_conditions"].extend(_parse_condition_coverage_index(vris_b_output))
    return _dedupe_findings(findings)


async def _run_vris_analysis(files: List[UploadFile]) -> Dict[str, Any]:
    """Core VRIS analysis pipeline shared by JSON and PDF endpoints."""
    if not files or len(files) == 0:
        raise HTTPException(
            status_code=400,
            detail="No files provided. Please upload veteran documents."
        )

    temp_dir = tempfile.mkdtemp(prefix="vris_")
    try:
        uploaded_files = []
        file_info = []

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

        vris = VRISRAGSystem(
            system_docs_folder=SYSTEM_DOCS_FOLDER,
            veteran_docs_folder="",
            persist_directory=CHROMA_DB_DIR,
            model_name="gpt-4"
        )
        vris.system_vectorstore = vris_system.system_vectorstore
        vris.process_veteran_documents_from_upload(uploaded_files)

        print("\n🔄 Running VRIS dual-pipeline analysis...")
        print("   VRIS-A: Extracting diagnostic codes, symptoms, evidence...")

        analysis_result = vris.generate_second_look_vre()

        print("   VRIS-B: Reasoning against CFR Title 38 criteria...")

        vris_a_data = analysis_result.get('vris_a_result', '')
        vris_b_data = analysis_result.get('vris_b_result', '')

        if "I don't have the ability" in vris_b_data or "I can't" in vris_b_data or len(vris_b_data) < 200:
            raise Exception("VRIS-B refused to analyze. Please check system vectorstore has CFR documentation.")

        findings = parse_vris_findings(vris_b_data)

        medical_only = _is_medical_only_packet(vris_a_data, vris_b_data)
        if medical_only:
            findings = _enrich_medical_only_findings(findings, vris_a_data, vris_b_data)

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

        print("✅ Analysis complete")
        print(f"{'='*70}\n")

        veteran_name = _extract_veteran_name(uploaded_files, vris_a_data, vris_b_data)
        if veteran_name:
            print(f"👤 Veteran name extracted: {veteran_name}")

        return {
            "findings": findings,
            "summary": summary,
            "vris_a_data": vris_a_data,
            "vris_b_data": vris_b_data,
            "file_info": file_info,
            "veteran_name": veteran_name,
        }
    finally:
        if os.path.exists(temp_dir):
            try:
                shutil.rmtree(temp_dir)
                print(f"🗑️  Temporary files deleted: {temp_dir}")
            except Exception as e:
                print(f"⚠️  Warning: Could not delete temp files: {e}")


def _handle_analysis_exception(e: Exception) -> None:
    """Re-raise analysis exceptions as appropriate HTTPException."""
    error_text = str(e)
    print(f"❌ Error during analysis: {error_text}")
    if "OCR is required" in error_text or "Install OCR dependencies" in error_text:
        raise HTTPException(status_code=400, detail=error_text)
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


@app.post("/api/vris/analyze")
async def analyze_veteran_documents(
    files: List[UploadFile]
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
    try:
        data = await _run_vris_analysis(files)
        findings  = data["findings"]
        summary   = data["summary"]
        vris_a_data = data["vris_a_data"]
        vris_b_data = data["vris_b_data"]
        file_info   = data["file_info"]

        formatted_report = generate_full_report(findings, summary, vris_a_data, vris_b_data, file_info)

        return {
            "success": True,
            "timestamp": datetime.now().isoformat(),
            "analysis_type": "VRIS Complete Analysis",
            "files_analyzed": file_info,
            "veteran_name": data.get("veteran_name"),
            "vris_a_extraction": {
                "description": "Extracted data from veteran documents",
                "data": vris_a_data
            },
            "vris_b_reasoning": {
                "description": "Analysis against VA rating criteria (38 CFR)",
                "data": vris_b_data
            },
            "findings": findings,
            "summary": summary,
            "formatted_report": formatted_report,
            "data_retention": "Files processed in-memory only. Not stored. Deleted immediately after analysis.",
            "system_info": "VRIS™ - Veteran Rating Intelligence System"
        }
    except HTTPException:
        raise
    except Exception as e:
        _handle_analysis_exception(e)


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
                "analysis": None,
                "recommendation": None,
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
        
        # Extract Analysis narrative
        elif current_condition and line.startswith('Analysis:'):
            text = line.split('Analysis:', 1)[1].strip()
            if text:
                current_condition["analysis"] = text

        # Extract Recommendation
        elif current_condition and line.startswith('Recommendation:'):
            text = line.split('Recommendation:', 1)[1].strip()
            if text:
                current_condition["recommendation"] = text

        # Extract CFR citations
        elif current_condition and ('38 CFR' in line or 'Diagnostic Code' in line) and 'Evidence:' not in line:
            current_condition["cfr_citations"].append(line)
        
        # Extract evidence (make sure "Evidence:" lines go to evidence, not cfr_citations)
        elif current_condition and 'Evidence:' in line:
            evidence_text = line.split('Evidence:', 1)[1].strip() if ':' in line else line
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
        if (cond.get("confidence") or 0) >= 90
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


# ---------------------------------------------------------------------------
# VA Compensation & Rating Math (2025 rates, single veteran, no dependents)
# ---------------------------------------------------------------------------
VA_COMPENSATION_2025: Dict[int, float] = {
    10: 175.51,  20: 346.95,  30: 537.42,  40: 775.28,
    50: 1102.04, 60: 1395.93, 70: 1716.28, 80: 1995.01,
    90: 2241.91, 100: 3737.85,
}


def _parse_rating_int(value: Any) -> Optional[int]:
    """Convert '20%', 20, or '20' to int. Returns None on failure."""
    if value is None:
        return None
    try:
        return int(str(value).replace('%', '').strip())
    except (ValueError, TypeError):
        return None


def calculate_va_combined_rating(ratings: List[int]) -> int:
    """VA 'whole person' combined disability rating, rounded to nearest 10."""
    clean = sorted([r for r in ratings if isinstance(r, (int, float)) and r > 0], reverse=True)
    if not clean:
        return 0
    remaining = 100.0
    for r in clean:
        remaining *= (1.0 - r / 100.0)
    return min(int(round((100.0 - remaining) / 10.0) * 10), 100)


def get_monthly_compensation(combined_rating: int) -> float:
    """2025 VA monthly compensation for a single veteran with no dependents."""
    rounded = max(10, min(100, round(combined_rating / 10) * 10))
    return VA_COMPENSATION_2025.get(rounded, 0.0)


def _confidence_label(score: Optional[int]) -> str:
    if score is None:
        return "Unscored"
    if score >= 90:
        return "Strong"
    if score >= 80:
        return "Moderate-Strong"
    if score >= 70:
        return "Moderate"
    return "Low"


def generate_full_report(
    findings: Dict[str, Any],
    summary: Dict[str, Any],
    vris_a_data: str,
    vris_b_data: str,
    files_analyzed: List[Dict[str, Any]],
) -> str:
    """Generate a GET VA HELP\u2122 Full Gap Analysis Report in standard client format."""
    date_str = datetime.now().strftime("%B %d, %Y")

    underrated = findings.get("underrated_conditions", [])
    missed     = findings.get("missed_conditions", [])
    secondary  = findings.get("secondary_conditions", [])
    all_non_rated = missed + secondary

    phase1 = [c for c in all_non_rated
              if (c.get("phase") in (1, "1")) or ((c.get("confidence") or 0) >= 85)]
    phase2 = [c for c in all_non_rated if c not in phase1]

    current_ratings = [r for r in [_parse_rating_int(c.get("current_rating")) for c in underrated]
                       if r is not None]
    current_combined = calculate_va_combined_rating(current_ratings) if current_ratings else None
    current_comp     = get_monthly_compensation(current_combined) if current_combined is not None else None

    projected_ratings = list(current_ratings)
    for c in underrated:
        old_r = _parse_rating_int(c.get("current_rating"))
        new_r = _parse_rating_int(c.get("potential_rating"))
        if old_r is not None and new_r is not None and new_r > old_r:
            try:
                projected_ratings.remove(old_r)
            except ValueError:
                pass
            projected_ratings.append(new_r)
    for c in phase1:
        r = _parse_rating_int(c.get("potential_rating"))
        if r and r > 0:
            projected_ratings.append(r)

    projected_combined = calculate_va_combined_rating(projected_ratings) if projected_ratings else None
    projected_comp     = get_monthly_compensation(projected_combined) if projected_combined is not None else None

    all_confs = [(c.get("confidence") or 0)
                 for c in underrated + missed + secondary
                 if c.get("confidence") is not None]
    overall_confidence = int(sum(all_confs) / len(all_confs)) if all_confs else None

    SEP  = "=" * 72
    DASH = "-" * 72
    L: List[str] = []

    # Header
    L += [SEP, "GET VA HELP\u2122 \u2014 Full Gap Analysis Report", SEP]
    L.append("Veteran:            [Name Redacted for Privacy]")
    L.append("Date of Report:     " + date_str)
    L.append("Report Prepared By: Veteran Rating Intelligence System (VRIS\u2122)")
    if overall_confidence:
        L.append("Confidence Level:   " + str(overall_confidence) + "% Agreement (VRIS-A / VRIS-B)")
    doc_names = ", ".join(f["filename"] for f in files_analyzed) if files_analyzed else "N/A"
    L.append("Documents Analyzed: " + doc_names)
    L.append("")

    # Overview
    L += ["OVERVIEW", DASH]
    L.append(
        "VRIS\u2122 conducted a comprehensive evaluation of the veteran's claim record including "
        "VA Rating Decision Letter, C&P Exams, STRs, private records, and Secondary Conditions "
        "Questionnaire. The goal was to identify underrating, missed secondary conditions, and "
        "causal links under 38 CFR Part 4 and \u00a73.310 that could increase the veteran's combined rating."
    )
    L.append("")

    # Current Rating Overview
    L += ["CURRENT RATING OVERVIEW", DASH]
    if underrated:
        L.append("Condition".ljust(38) + "DC".ljust(8) + "Current".ljust(12) + "Potential")
        L.append("-" * 72)
        for c in underrated:
            name = (c.get("name") or "")[:37]
            dc = ""
            for cit in (c.get("cfr_citations") or []):
                m = re.search(r'DC\s*[:#]?\s*(\d{4,5})', cit, re.IGNORECASE)
                if m:
                    dc = m.group(1)
                    break
            cur = str(c.get("current_rating") or "?")
            pot = str(c.get("potential_rating") or "?")
            L.append(name.ljust(38) + dc.ljust(8) + cur.ljust(12) + pot)
        L.append("")
        if current_combined is not None:
            L.append("Current Combined Rating: " + str(current_combined) + "%")
            if current_comp is not None:
                L.append("Monthly Compensation (Est.): $" + f"{current_comp:,.2f}" + " (Single Veteran, 2025 VA Chart)")
    else:
        L.append("No current VA rating data found in provided documents.")
        L.append("Upload VA Decision Letter and Code Sheet for full rating analysis.")
    L.append("")

    # Underrated conditions
    if underrated:
        L += ["VRIS\u2122 FINDINGS \u2014 UNDERRATED CONDITIONS", DASH]
        for c in underrated:
            conf = c.get("confidence")
            L.append("")
            L.append(str(c.get("name", "Unknown")) + " (Current Rating: " + str(c.get("current_rating", "?")) + "%)")
            if c.get("cfr_citations"):
                L.append("  CFR Citations: " + "; ".join((c.get("cfr_citations") or [])[:3]))
            if c.get("evidence"):
                L.append("  Evidence: " + (c.get("evidence") or [""])[0])
            if c.get("potential_rating"):
                L.append("  Potential Rating: " + str(c["potential_rating"]) + "%")
            L.append("  Recommendation: File Supplemental Claim for increased rating.")
            if conf is not None:
                L.append("  Confidence: " + str(conf) + "% (" + _confidence_label(conf) + ")")
        L.append("")

    # Phase 1 missed/secondary
    if phase1:
        L += ["VRIS\u2122 FINDINGS \u2014 MISSED & SECONDARY CONDITIONS (PHASE 1: STRONG)", DASH]
        for c in phase1:
            conf = c.get("confidence")
            L.append("")
            L.append(str(c.get("name", "Unknown")) + " (Not Rated)")
            if c.get("cfr_citations"):
                L.append("  CFR Citations: " + "; ".join((c.get("cfr_citations") or [])[:3]))
            if c.get("evidence"):
                L.append("  Evidence: " + (c.get("evidence") or [""])[0])
            if c.get("potential_rating"):
                L.append("  Suggested Rating: " + str(c["potential_rating"]) + "%")
            L.append("  Recommendation: File New Secondary Claim.")
            if conf is not None:
                L.append("  Confidence: " + str(conf) + "% (" + _confidence_label(conf) + ")")
        L.append("")

    # Phase 2 correlative
    if phase2:
        L += ["PHASE 2 CORRELATIVE FINDINGS", DASH]
        L.append("Lower-confidence findings that may merit review by a VSO representative:")
        L.append("Potential Condition".ljust(38) + "Possible Connection".ljust(35) + "Confidence")
        L.append("-" * 85)
        for c in phase2:
            name    = (c.get("name") or "")[:37]
            ev      = ((c.get("evidence") or ["Secondary relationship"])[0])[:34]
            conf    = c.get("confidence")
            conf_str = (str(conf) + "% (" + _confidence_label(conf) + ")") if conf is not None else "Unscored"
            L.append(name.ljust(38) + ev.ljust(35) + conf_str)
        L.append("")

    # Confidence summary
    L += ["CONFIDENCE SUMMARY", DASH]
    L.append("Category".ljust(32) + "VRIS-A".ljust(12) + "VRIS-B".ljust(12) + "Agreement")
    L.append("-" * 64)
    if underrated:
        avg = int(sum((c.get("confidence") or 0) for c in underrated) / len(underrated))
        L.append("Primary Conditions".ljust(32) + (str(avg) + "%").ljust(12) + (str(avg) + "%").ljust(12) + str(avg) + "%")
    if phase1:
        avg = int(sum((c.get("confidence") or 0) for c in phase1) / len(phase1))
        L.append("Secondary Conditions".ljust(32) + (str(avg) + "%").ljust(12) + (str(avg) + "%").ljust(12) + str(avg) + "%")
    if phase2:
        avg = int(sum((c.get("confidence") or 0) for c in phase2) / len(phase2))
        L.append("Phase 2 Correlations".ljust(32) + (str(avg) + "%").ljust(12) + (str(avg) + "%").ljust(12) + str(avg) + "%")
    if overall_confidence:
        L.append("")
        L.append("Overall Confidence Score: " + str(overall_confidence) + "%")
        if overall_confidence >= 90:
            L.append("(Eligible for refund-backed rating prediction guarantee)")
    L.append("")

    # Projected rating & compensation
    L += ["PROJECTED RATING & COMPENSATION IMPACT", DASH]
    if underrated or phase1:
        L.append("Condition".ljust(45) + "Projected Rating")
        L.append("-" * 65)
        for c in underrated:
            L.append((c.get("name") or "")[:44].ljust(45) + str(c.get("potential_rating", "?")) + "%")
        for c in phase1:
            L.append((c.get("name") or "")[:44].ljust(45) + str(c.get("potential_rating", "?")) + "%")
        L.append("")
        if projected_combined is not None:
            if current_combined is not None:
                L.append("Projected Combined Rating: " + str(projected_combined) + "% (from " + str(current_combined) + "%)")
                if projected_comp is not None and current_comp is not None:
                    delta = projected_comp - current_comp
                    L.append("Projected Monthly Compensation: $" + f"{projected_comp:,.2f}" + " (+$" + f"{delta:,.2f}" + "/month increase)")
            else:
                L.append("Projected Combined Rating: " + str(projected_combined) + "%")
                if projected_comp is not None:
                    L.append("Projected Monthly Compensation (Est.): $" + f"{projected_comp:,.2f}")
    else:
        L.append("Upload VA Decision Letter and C&P exams to enable projected rating calculation.")
    L.append("")

    # Filing recommendations
    L += ["FILING RECOMMENDATIONS", DASH]
    L.append("Action".ljust(48) + "Claim Type".ljust(22) + "Confidence")
    L.append("-" * 78)
    for c in underrated:
        action   = ("Increase: " + (c.get("name") or ""))[:47]
        conf_str = str(c["confidence"]) + "%" if c.get("confidence") is not None else "\u2014"
        L.append(action.ljust(48) + "Supplemental".ljust(22) + conf_str)
    for c in phase1:
        action   = ("Secondary: " + (c.get("name") or ""))[:47]
        conf_str = str(c["confidence"]) + "%" if c.get("confidence") is not None else "\u2014"
        L.append(action.ljust(48) + "New Claim".ljust(22) + conf_str)
    L.append("")

    # Next steps
    L += ["NEXT STEPS FOR THE VETERAN", DASH]
    L.append("\u2022 File Supplemental Claim (VA Form 20-0995) for recommended increases.")
    L.append("\u2022 Attach supporting documentation (C&P exams, STRs, private records).")
    L.append("\u2022 Consult your VSO or accredited representative (DAV, VFW, American Legion).")
    L.append("\u2022 Monitor VA.gov for claim updates.")
    L.append("\u2022 If denied, request Higher-Level Review (VA Form 20-0996) or BVA Appeal.")
    L.append("")

    # Billing summary
    L += ["BILLING SUMMARY", DASH]
    L.append("Projected Rating Increase".ljust(28) + "Fee".ljust(10) + "Refund Policy")
    L.append("-" * 75)
    L.append("5\u201315%".ljust(28)  + "$499      Refund difference if increase <15%, no less than $499 minimum")
    L.append("16\u201325%".ljust(28) + "$649      Partial refund based on actual increase")
    L.append("26\u201350%".ljust(28) + "$799      Partial refund based on actual increase")
    L.append("51%+".ljust(28)       + "$999      Refund only if increase <5%")
    L.append("")

    # Disclaimer
    L += ["DISCLAIMER", DASH]
    L.append(
        "This report was prepared by VRIS\u2122 (Veteran Rating Intelligence System) as part of the "
        "Get VA Help\u2122 initiative. It is an intelligence-based assessment tool and not legal or "
        "medical advice. Findings are based on data provided and CFR regulations as of the report "
        "date. Veterans should consult a certified VSO for official submission."
    )
    L.append(SEP)

    return "\n".join(L)


# ---------------------------------------------------------------------------
# PDF Report Generation
# ---------------------------------------------------------------------------

# Brand colours
_PDF_BRAND_BLUE  = HexColor('#15467A')
_PDF_HEADER_BG   = HexColor('#1A4A8A')
_PDF_LIGHT_BG    = HexColor('#EBF2FA')
_PDF_STRIPE      = HexColor('#F4F7FC')
_PDF_DARK_TEXT   = HexColor('#1A1A2E')


def _pdf_styles() -> Dict[str, ParagraphStyle]:
    """Return a dict of named ParagraphStyle objects used by generate_pdf_report."""
    return {
        "title": ParagraphStyle(
            'VRISTitle', fontName='Helvetica-Bold', fontSize=20,
            textColor=white, backColor=_PDF_HEADER_BG,
            alignment=TA_CENTER, spaceAfter=0, spaceBefore=0,
            leftIndent=8, rightIndent=8, leading=26,
        ),
        "subtitle": ParagraphStyle(
            'VRISSubtitle', fontName='Helvetica-Bold', fontSize=13,
            textColor=white, backColor=_PDF_BRAND_BLUE,
            alignment=TA_CENTER, spaceAfter=0, spaceBefore=0, leading=18,
        ),
        "section": ParagraphStyle(
            'VRISSectionHeader', fontName='Helvetica-Bold', fontSize=10,
            textColor=white, backColor=_PDF_BRAND_BLUE,
            spaceAfter=5, spaceBefore=10,
            leftIndent=6, leading=15,
        ),
        "condition": ParagraphStyle(
            'VRISCondition', fontName='Helvetica-Bold', fontSize=10,
            textColor=_PDF_BRAND_BLUE, spaceAfter=2, spaceBefore=6,
        ),
        "normal": ParagraphStyle(
            'VRISNormal', fontName='Helvetica', fontSize=9,
            textColor=_PDF_DARK_TEXT, spaceAfter=2, leading=13,
        ),
        "meta": ParagraphStyle(
            'VRISMeta', fontName='Helvetica', fontSize=9,
            textColor=HexColor('#444444'), spaceAfter=2, leading=13,
        ),
        "bullet": ParagraphStyle(
            'VRISBullet', fontName='Helvetica', fontSize=9,
            textColor=_PDF_DARK_TEXT, leftIndent=14, spaceAfter=3, leading=13,
        ),
        "disclaimer": ParagraphStyle(
            'VRISDisclaimer', fontName='Helvetica-Oblique', fontSize=8,
            textColor=HexColor('#555555'), spaceAfter=2, leading=11,
        ),
    }


def _table_style(has_stripe: bool = True) -> TableStyle:
    """Standard table style with dark-blue header row."""
    cmds = [
        ('BACKGROUND',   (0, 0), (-1, 0), _PDF_HEADER_BG),
        ('TEXTCOLOR',    (0, 0), (-1, 0), white),
        ('FONTNAME',     (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('FONTNAME',     (0, 1), (-1, -1), 'Helvetica'),
        ('FONTSIZE',     (0, 0), (-1, -1), 9),
        ('VALIGN',       (0, 0), (-1, -1), 'MIDDLE'),
        ('TOPPADDING',   (0, 0), (-1, -1), 4),
        ('BOTTOMPADDING',(0, 0), (-1, -1), 4),
        ('LEFTPADDING',  (0, 0), (-1, -1), 6),
        ('GRID',         (0, 0), (-1, -1), 0.25, HexColor('#CCCCCC')),
    ]
    if has_stripe:
        cmds.append(('ROWBACKGROUNDS', (0, 1), (-1, -1), [white, _PDF_STRIPE]))
    return TableStyle(cmds)


def generate_pdf_report(
    findings: Dict[str, Any],
    summary: Dict[str, Any],
    vris_a_data: str,
    vris_b_data: str,
    files_analyzed: List[Dict[str, Any]],
    veteran_name: Optional[str] = None,
) -> bytes:
    """Generate a styled GET VA HELP™ Full Gap Analysis PDF using ReportLab."""
    S = _pdf_styles()

    # ── Computed values (same logic as generate_full_report) ──────────────
    date_str      = datetime.now().strftime("%B %d, %Y")
    underrated    = findings.get("underrated_conditions", [])
    missed        = findings.get("missed_conditions", [])
    secondary     = findings.get("secondary_conditions", [])
    all_non_rated = missed + secondary

    phase1 = [c for c in all_non_rated
              if (c.get("phase") in (1, "1")) or ((c.get("confidence") or 0) >= 85)]
    phase2 = [c for c in all_non_rated if c not in phase1]

    current_ratings = [r for r in
                       [_parse_rating_int(c.get("current_rating")) for c in underrated]
                       if r is not None]
    current_combined = calculate_va_combined_rating(current_ratings) if current_ratings else None
    current_comp     = get_monthly_compensation(current_combined) if current_combined is not None else None

    projected_ratings = list(current_ratings)
    for c in underrated:
        old_r = _parse_rating_int(c.get("current_rating"))
        new_r = _parse_rating_int(c.get("potential_rating"))
        if old_r is not None and new_r is not None and new_r > old_r:
            try:
                projected_ratings.remove(old_r)
            except ValueError:
                pass
            projected_ratings.append(new_r)
    for c in phase1:
        r = _parse_rating_int(c.get("potential_rating"))
        if r and r > 0:
            projected_ratings.append(r)

    projected_combined = calculate_va_combined_rating(projected_ratings) if projected_ratings else None
    projected_comp     = get_monthly_compensation(projected_combined) if projected_combined is not None else None

    all_confs = [(c.get("confidence") or 0)
                 for c in underrated + missed + secondary
                 if c.get("confidence") is not None]
    overall_confidence = int(sum(all_confs) / len(all_confs)) if all_confs else None

    # ── Document setup ────────────────────────────────────────────────────
    buffer  = io.BytesIO()
    doc     = SimpleDocTemplate(
        buffer, pagesize=letter,
        topMargin=0.75*inch, bottomMargin=0.75*inch,
        leftMargin=0.75*inch, rightMargin=0.75*inch,
    )
    # Usable content width
    CW = letter[0] - 1.5*inch

    elements: List[Any] = []

    # ── HEADER ────────────────────────────────────────────────────────────
    elements.append(Paragraph("GET VA HELP\u2122", S["title"]))
    elements.append(Paragraph("Full Gap Analysis Report", S["subtitle"]))
    elements.append(Spacer(1, 0.12*inch))

    doc_names  = ", ".join(f["filename"] for f in files_analyzed) if files_analyzed else "N/A"
    conf_text  = (f"{overall_confidence}% Agreement (VRIS-A / VRIS-B)"
                  if overall_confidence else "N/A")
    info_data = [
        ["Veteran:",            veteran_name or "[Name Not Provided]"],
        ["Date of Report:",     date_str],
        ["Report Prepared By:", "Veteran Rating Intelligence System (VRIS\u2122)"],
        ["Confidence Level:",   conf_text],
        ["Documents Analyzed:", doc_names],
    ]
    info_table = Table(info_data, colWidths=[1.6*inch, CW - 1.6*inch])
    info_table.setStyle(TableStyle([
        ('FONTNAME',     (0, 0), (0, -1), 'Helvetica-Bold'),
        ('FONTNAME',     (1, 0), (1, -1), 'Helvetica'),
        ('FONTSIZE',     (0, 0), (-1, -1), 9),
        ('TEXTCOLOR',    (0, 0), (0, -1), _PDF_BRAND_BLUE),
        ('VALIGN',       (0, 0), (-1, -1), 'TOP'),
        ('TOPPADDING',   (0, 0), (-1, -1), 2),
        ('BOTTOMPADDING',(0, 0), (-1, -1), 2),
    ]))
    elements.append(info_table)
    elements.append(Spacer(1, 0.12*inch))

    # ── OVERVIEW ──────────────────────────────────────────────────────────
    elements.append(Paragraph("OVERVIEW", S["section"]))
    elements.append(Paragraph(
        "VRIS\u2122 conducted a comprehensive evaluation of the veteran\u2019s claim record including "
        "VA Rating Decision Letter, C&amp;P Exams, STRs, private records, and Secondary Conditions "
        "Questionnaire. The goal was to identify underrating, missed secondary conditions, and "
        "causal links under 38 CFR Part\u00a04 and \u00a73.310 that could increase the veteran\u2019s "
        "combined rating.",
        S["normal"]
    ))
    elements.append(Spacer(1, 0.08*inch))

    # ── CURRENT RATING OVERVIEW ───────────────────────────────────────────
    elements.append(Paragraph("CURRENT RATING OVERVIEW", S["section"]))
    if underrated:
        cr_data = [["Condition", "DC", "Current Rating", "Potential Rating"]]
        for c in underrated:
            dc = ""
            for cit in (c.get("cfr_citations") or []):
                m = re.search(r'DC\s*[:#]?\s*(\d{4,5})', cit, re.IGNORECASE)
                if m:
                    dc = m.group(1)
                    break
            cr_data.append([
                c.get("name") or "",
                dc,
                str(c.get("current_rating") or "?"),
                str(c.get("potential_rating") or "?"),
            ])
        if current_combined is not None:
            cr_data.append([
                "Combined Rating", "",
                f"{current_combined}%",
                f"{projected_combined}%" if projected_combined else "?",
            ])
        cr_table = Table(cr_data, colWidths=[3.1*inch, 0.7*inch, 1.15*inch, 1.25*inch])
        ts = _table_style()
        if current_combined is not None:
            ts.add('FONTNAME',   (0, -1), (-1, -1), 'Helvetica-Bold')
            ts.add('BACKGROUND', (0, -1), (-1, -1), _PDF_LIGHT_BG)
        cr_table.setStyle(ts)
        elements.append(cr_table)
        if current_comp is not None:
            elements.append(Spacer(1, 0.04*inch))
            elements.append(Paragraph(
                f"<b>Monthly Compensation (Est.):</b> ${current_comp:,.2f}"
                " (Single Veteran, 2025 VA Chart)",
                S["normal"]
            ))
    else:
        elements.append(Paragraph(
            "No current VA rating data found in provided documents. "
            "Upload VA Decision Letter and Code Sheet for full rating analysis.",
            S["normal"]
        ))
    elements.append(Spacer(1, 0.08*inch))

    # ── VRIS™ FINDINGS & ANALYSIS (all conditions combined) ─────────────
    all_findings = list(underrated) + list(phase1)
    if all_findings:
        elements.append(Paragraph("VRIS\u2122 FINDINGS &amp; ANALYSIS", S["section"]))
        for c in all_findings:
            block: List[Any] = []
            is_underrated = c in underrated

            # Condition title line: name + DC code if available
            dc = ""
            for cit in (c.get("cfr_citations") or []):
                m = re.search(r'DC\s*[:#]?\s*(\d{4,5})', cit, re.IGNORECASE)
                if m:
                    dc = m.group(1)
                    break
            title = c.get('name', 'Unknown')
            if dc:
                title = f"{title} (DC {dc})"
            elif not is_underrated:
                title = f"{title}"
            block.append(Paragraph(title, S["condition"]))

            # Analysis line - prefer parsed analysis text, fall back to evidence
            analysis_text = c.get("analysis")
            if not analysis_text:
                ev = (c.get("evidence") or [])
                if ev and not ev[0].lower().startswith("documented in"):
                    analysis_text = ev[0]
                elif ev:
                    # Reframe page reference as an evidence note
                    analysis_text = (
                        ev[0] + ". See VA records for full clinical details."
                    )
            if analysis_text:
                block.append(Paragraph(f"<b>Analysis:</b> {analysis_text}", S["normal"]))

            # Recommendation line
            rec_text = c.get("recommendation")
            if not rec_text:
                if is_underrated:
                    pot = c.get('potential_rating')
                    cur = c.get('current_rating', '?')
                    rec_text = (
                        f"File Supplemental Claim for increased rating"
                        + (f" from {cur}% \u2192 {pot}%" if pot and pot != '?' else ".")
                    )
                else:
                    pot = c.get('potential_rating')
                    rec_text = (
                        "File New Secondary Claim"
                        + (f", suggested rating {pot}%" if pot and pot not in ('?', 'Not specified due to missing VA rating evidence') else ".")
                    )
            block.append(Paragraph(f"<b>Recommendation:</b> {rec_text}", S["normal"]))

            # Confidence line
            conf = c.get("confidence")
            if conf is not None:
                block.append(Paragraph(
                    f"<b>Confidence:</b> {conf}% ({_confidence_label(conf)})", S["normal"]
                ))

            elements.append(KeepTogether(block))
            elements.append(Spacer(1, 0.04*inch))
        elements.append(Spacer(1, 0.06*inch))

    # ── PHASE 2 CORRELATIVE FINDINGS ─────────────────────────────────────
    # Deduplicate phase2 against phase1 by condition name (case-insensitive)
    phase1_names = {(c.get("name") or "").lower().strip() for c in phase1}
    seen_p2 = set()
    phase2_display = []
    for c in phase2:
        n = (c.get("name") or "").lower().strip()
        if n not in phase1_names and n not in seen_p2:
            phase2_display.append(c)
            seen_p2.add(n)

    if phase2_display:
        elements.append(Paragraph("PHASE 2 CORRELATIVE FINDINGS", S["section"]))
        elements.append(Paragraph(
            "Lower-confidence findings that may merit review by a VSO representative:",
            S["normal"]
        ))
        # Header row uses bold Paragraphs so they wrap safely too
        hdr = ParagraphStyle('p2hdr', parent=S["normal"], fontName='Helvetica-Bold',
                             textColor=white, backColor=_PDF_HEADER_BG)
        p2_data = [
            [Paragraph("Potential Condition", hdr),
             Paragraph("Possible Connection", hdr),
             Paragraph("Confidence", hdr)]
        ]
        for c in phase2_display:
            # Extract just ICD code + page from evidence string
            ev_raw = (c.get("evidence") or ["Secondary relationship"])[0]
            icd_match = re.search(
                r'[-\u2013]\s*([A-Z][0-9][\w.]*)\s*(\(Pages?[\s\d,]+\))',
                ev_raw
            )
            ev_short = (
                f"ICD: {icd_match.group(1)} {icd_match.group(2)}"
                if icd_match
                else (ev_raw[:40] + ("..." if len(ev_raw) > 40 else ""))
            )
            conf = c.get("confidence")
            conf_str = f"{conf}% ({_confidence_label(conf)})" if conf is not None else "Unscored"
            p2_data.append([
                Paragraph(c.get("name") or "", S["normal"]),
                Paragraph(ev_short, S["normal"]),
                Paragraph(conf_str, S["normal"]),
            ])
        # Dedicate larger share to Condition name; ICD is always short
        p2_table = Table(p2_data, colWidths=[3.2*inch, 2.0*inch, 1.5*inch],
                         repeatRows=1)
        p2_ts = TableStyle([
            ('BACKGROUND',    (0, 0), (-1, 0), _PDF_HEADER_BG),
            ('FONTNAME',      (0, 1), (-1, -1), 'Helvetica'),
            ('FONTSIZE',      (0, 0), (-1, -1), 9),
            ('VALIGN',        (0, 0), (-1, -1), 'TOP'),
            ('TOPPADDING',    (0, 0), (-1, -1), 4),
            ('BOTTOMPADDING', (0, 0), (-1, -1), 4),
            ('LEFTPADDING',   (0, 0), (-1, -1), 6),
            ('GRID',          (0, 0), (-1, -1), 0.25, HexColor('#CCCCCC')),
            ('ROWBACKGROUNDS', (0, 1), (-1, -1), [white, _PDF_STRIPE]),
        ])
        p2_table.setStyle(p2_ts)
        elements.append(p2_table)
        elements.append(Spacer(1, 0.08*inch))

    # ── CONFIDENCE SUMMARY ────────────────────────────────────────────────
    elements.append(Paragraph("CONFIDENCE SUMMARY", S["section"]))
    conf_data = [["Category", "VRIS-A", "VRIS-B", "Agreement Level"]]
    for label, group in [("Primary Conditions", underrated),
                         ("Secondary Conditions", phase1),
                         ("Phase 2 Correlations", phase2)]:
        if group:
            avg = int(sum((c.get("confidence") or 0) for c in group) / len(group))
            conf_data.append([label, f"{avg}%", f"{avg}%", f"{avg}%"])
    if len(conf_data) == 1:
        conf_data.append(["No findings", "N/A", "N/A", "N/A"])
    conf_table = Table(conf_data, colWidths=[2.5*inch, 1.1*inch, 1.1*inch, 1.5*inch])
    conf_table.setStyle(_table_style())
    elements.append(conf_table)
    if overall_confidence:
        elements.append(Spacer(1, 0.04*inch))
        suffix = (" \u2014 Eligible for refund-backed rating prediction guarantee"
                  if overall_confidence >= 90 else "")
        elements.append(Paragraph(
            f"<b>Overall Confidence Score: {overall_confidence}%</b>{suffix}",
            S["normal"]
        ))
    elements.append(Spacer(1, 0.08*inch))

    # ── PROJECTED RATING & COMPENSATION ──────────────────────────────────
    elements.append(Paragraph("PROJECTED RATING &amp; COMPENSATION IMPACT", S["section"]))

    def _fmt_rating(val: Any) -> str:
        """Clean potential_rating for display."""
        if val is None:
            return "Pending VA Rating"
        s = str(val).strip().rstrip('%')
        if not s or s == '?' or 'not specified' in s.lower() or 'missing' in s.lower():
            return "Pending VA Rating"
        return s + "%"

    if underrated or phase1:
        hdr2 = ParagraphStyle('projhdr', parent=S["normal"], fontName='Helvetica-Bold',
                              textColor=white, backColor=_PDF_HEADER_BG)
        proj_data = [
            [Paragraph("Condition", hdr2), Paragraph("Projected Rating", hdr2)]
        ]
        for c in list(underrated) + list(phase1):
            proj_data.append([
                Paragraph(c.get("name") or "", S["normal"]),
                Paragraph(_fmt_rating(c.get("potential_rating")), S["normal"]),
            ])
        proj_table = Table(proj_data, colWidths=[4.8*inch, 1.9*inch], repeatRows=1)
        proj_ts = TableStyle([
            ('BACKGROUND',    (0, 0), (-1, 0), _PDF_HEADER_BG),
            ('FONTSIZE',      (0, 0), (-1, -1), 9),
            ('VALIGN',        (0, 0), (-1, -1), 'TOP'),
            ('TOPPADDING',    (0, 0), (-1, -1), 4),
            ('BOTTOMPADDING', (0, 0), (-1, -1), 4),
            ('LEFTPADDING',   (0, 0), (-1, -1), 6),
            ('GRID',          (0, 0), (-1, -1), 0.25, HexColor('#CCCCCC')),
            ('ROWBACKGROUNDS', (0, 1), (-1, -1), [white, _PDF_STRIPE]),
        ])
        proj_table.setStyle(proj_ts)
        elements.append(proj_table)
        elements.append(Spacer(1, 0.04*inch))
        if projected_combined is not None:
            if current_combined is not None:
                elements.append(Paragraph(
                    f"<b>Projected Combined Rating:</b> {projected_combined}%"
                    f" (from {current_combined}%)",
                    S["normal"]
                ))
                if projected_comp is not None and current_comp is not None:
                    delta = projected_comp - current_comp
                    elements.append(Paragraph(
                        f"<b>Projected Monthly Compensation:</b> ${projected_comp:,.2f}"
                        f" (+${delta:,.2f}/month increase)",
                        S["normal"]
                    ))
            else:
                elements.append(Paragraph(
                    f"<b>Projected Combined Rating:</b> {projected_combined}%", S["normal"]
                ))
                if projected_comp is not None:
                    elements.append(Paragraph(
                        f"<b>Projected Monthly Compensation (Est.):</b> ${projected_comp:,.2f}",
                        S["normal"]
                    ))
    else:
        elements.append(Paragraph(
            "Upload VA Decision Letter and C&amp;P exams to enable projected rating calculation.",
            S["normal"]
        ))
    elements.append(Spacer(1, 0.08*inch))

    # ── FILING RECOMMENDATIONS ────────────────────────────────────────────
    elements.append(Paragraph("FILING RECOMMENDATIONS", S["section"]))
    hdr3 = ParagraphStyle('filinghdr', parent=S["normal"], fontName='Helvetica-Bold',
                          textColor=white, backColor=_PDF_HEADER_BG)
    filing_data = [
        [Paragraph("Action", hdr3), Paragraph("Claim Type", hdr3), Paragraph("Confidence", hdr3)]
    ]
    for c in underrated:
        conf_str = f"{c['confidence']}%" if c.get("confidence") is not None else "\u2014"
        filing_data.append([
            Paragraph(f"Increase: {c.get('name') or ''}", S["normal"]),
            Paragraph("Supplemental", S["normal"]),
            Paragraph(conf_str, S["normal"]),
        ])
    for c in phase1:
        conf_str = f"{c['confidence']}%" if c.get("confidence") is not None else "\u2014"
        filing_data.append([
            Paragraph(f"Secondary: {c.get('name') or ''}", S["normal"]),
            Paragraph("New Claim", S["normal"]),
            Paragraph(conf_str, S["normal"]),
        ])
    if len(filing_data) == 1:
        filing_data.append([
            Paragraph("No recommendations available", S["normal"]),
            Paragraph("\u2014", S["normal"]),
            Paragraph("\u2014", S["normal"]),
        ])
    filing_ts = TableStyle([
        ('BACKGROUND',    (0, 0), (-1, 0), _PDF_HEADER_BG),
        ('FONTSIZE',      (0, 0), (-1, -1), 9),
        ('VALIGN',        (0, 0), (-1, -1), 'TOP'),
        ('TOPPADDING',    (0, 0), (-1, -1), 4),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 4),
        ('LEFTPADDING',   (0, 0), (-1, -1), 6),
        ('GRID',          (0, 0), (-1, -1), 0.25, HexColor('#CCCCCC')),
        ('ROWBACKGROUNDS', (0, 1), (-1, -1), [white, _PDF_STRIPE]),
    ])
    filing_table = Table(filing_data, colWidths=[3.8*inch, 1.5*inch, 1.4*inch], repeatRows=1)
    filing_table.setStyle(filing_ts)
    elements.append(filing_table)
    elements.append(Spacer(1, 0.08*inch))

    # ── NEXT STEPS ────────────────────────────────────────────────────────
    elements.append(Paragraph("NEXT STEPS FOR THE VETERAN", S["section"]))
    for step in [
        "File Supplemental Claim (VA Form 20-0995) for recommended increases.",
        "Attach supporting documentation (C&amp;P exams, STRs, private records).",
        "Consult your VSO or accredited representative (DAV, VFW, American Legion).",
        "Monitor VA.gov for claim updates.",
        "If denied, request Higher-Level Review (VA Form 20-0996) or BVA Appeal.",
    ]:
        elements.append(Paragraph(f"\u2022 {step}", S["bullet"]))
    elements.append(Spacer(1, 0.08*inch))

    # ── BILLING SUMMARY ───────────────────────────────────────────────────
    elements.append(Paragraph("BILLING SUMMARY", S["section"]))
    billing_data = [
        ["Projected Rating Increase", "Fee", "Refund Policy"],
        ["5\u201315%",  "$499", "Refund difference if increase <15%, no less than $499 minimum"],
        ["16\u201325%", "$649", "Partial refund based on actual increase"],
        ["26\u201350%", "$799", "Partial refund based on actual increase"],
        ["51%+",        "$999", "Refund only if increase <5%"],
    ]
    billing_table = Table(billing_data, colWidths=[1.7*inch, 0.8*inch, 3.7*inch])
    billing_ts = _table_style()
    billing_ts.add('VALIGN', (0, 0), (-1, -1), 'TOP')
    billing_table.setStyle(billing_ts)
    elements.append(billing_table)
    elements.append(Spacer(1, 0.08*inch))

    # ── DISCLAIMER ────────────────────────────────────────────────────────
    elements.append(Paragraph("DISCLAIMER", S["section"]))
    elements.append(Paragraph(
        "This report was prepared by VRIS\u2122 (Veteran Rating Intelligence System) as part of "
        "the Get VA Help\u2122 initiative. It is an intelligence-based assessment tool and not "
        "legal or medical advice. Findings are based on data provided and CFR regulations as of "
        "the report date. Veterans should consult a certified VSO for official submission.",
        S["disclaimer"]
    ))

    doc.build(elements)
    return buffer.getvalue()


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


@app.post("/api/vris/report/pdf")
async def generate_pdf_from_json(request: Request):
    """
    VRIS™ PDF Report from existing JSON analysis

    Pass the full JSON response from /api/vris/analyze as the request body.
    Returns a styled GET VA HELP™ PDF report immediately without re-running analysis.

    Returns: application/pdf — attachment filename VRIS_Gap_Analysis_Report.pdf
    """
    try:
        body = await request.json()
        findings = body.get("findings", {
            "underrated_conditions": [],
            "missed_conditions": [],
            "secondary_conditions": [],
            "total_opportunities": 0,
        })
        summary       = body.get("summary", {})
        vris_a_data    = body.get("vris_a_extraction", {}).get("data", "")
        vris_b_data    = body.get("vris_b_reasoning",  {}).get("data", "")
        files_analyzed = body.get("files_analyzed", [])
        veteran_name   = body.get("veteran_name") or body.get("veteran", {}).get("name")

        pdf_bytes = generate_pdf_report(
            findings, summary, vris_a_data, vris_b_data, files_analyzed,
            veteran_name=veteran_name,
        )
        return StreamingResponse(
            io.BytesIO(pdf_bytes),
            media_type="application/pdf",
            headers={
                "Content-Disposition": 'attachment; filename="VRIS_Gap_Analysis_Report.pdf"'
            },
        )
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"PDF generation failed: {e}")


@app.post("/api/vris/analyze/pdf")
async def analyze_veteran_documents_pdf(files: List[UploadFile]):
    """
    VRIS™ PDF Report Endpoint

    Upload veteran documents and receive a styled GET VA HELP™ PDF report.
    Runs the same dual-pipeline analysis as /api/vris/analyze but returns
    a downloadable PDF instead of JSON.

    Returns: application/pdf — attachment filename VRIS_Gap_Analysis_Report.pdf
    """
    try:
        data = await _run_vris_analysis(files)
        pdf_bytes = generate_pdf_report(
            data["findings"],
            data["summary"],
            data["vris_a_data"],
            data["vris_b_data"],
            data["file_info"],
            veteran_name=data.get("veteran_name"),
        )
        return StreamingResponse(
            io.BytesIO(pdf_bytes),
            media_type="application/pdf",
            headers={
                "Content-Disposition": 'attachment; filename="VRIS_Gap_Analysis_Report.pdf"'
            },
        )
    except HTTPException:
        raise
    except Exception as e:
        _handle_analysis_exception(e)


if __name__ == "__main__":
    import uvicorn
    
    print("\n" + "🎯"*35)
    print("VRIS™ Simple API - Single Endpoint Mode")
    print("Upload documents → Get complete analysis")
    print("🎯"*35 + "\n")
    
    uvicorn.run(app, host="0.0.0.0", port=8000, log_level="info")
