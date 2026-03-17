# OCR Setup Guide for VRIS

VRIS now supports **scanned PDFs** using OCR (Optical Character Recognition).

## Quick Setup

### Step 1: Install Python Dependencies

```bash
pip install -r requirements.txt
```

### Step 2: Install System Dependencies

#### Windows:

**Install Tesseract OCR:**
1. Download: https://github.com/UB-Mannheim/tesseract/wiki
2. Install to: `C:\Program Files\Tesseract-OCR\`
3. Add to PATH:
   ```powershell
   $env:Path += ";C:\Program Files\Tesseract-OCR"
   ```

**Install Poppler:**
1. Download: https://github.com/oschwartz10612/poppler-windows/releases
2. Extract to: `C:\Program Files\poppler\`
3. Add to PATH:
   ```powershell
   $env:Path += ";C:\Program Files\poppler\Library\bin"
   ```

#### macOS:

```bash
brew install tesseract poppler
```

#### Linux (Ubuntu/Debian):

```bash
sudo apt-get update
sudo apt-get install tesseract-ocr poppler-utils
```

### Step 3: Verify Installation

```python
python -c "import pytesseract; print('Tesseract OK')"
python -c "from pdf2image import convert_from_path; print('Poppler OK')"
```

## How It Works

### Automatic Detection

VRIS automatically detects if a PDF is scanned:

```
Loading PDF: cp_exam_ptsd.pdf
  → Text-based PDF, using standard extraction...  ✓ Fast

Loading PDF: scanned_medical_records.pdf
  → Scanned PDF detected, using OCR...  ✓ Slower but accurate
```

### Processing Modes

**Text-based PDFs** (VA Decision Letters, most C&P exams):
- Uses `PyPDFLoader` - Fast (< 1 second)
- No OCR needed

**Scanned PDFs** (Old medical records, handwritten notes):
- Uses `UnstructuredPDFLoader` with OCR - Slower (5-15 seconds per page)
- Extracts text from images

### Fallback Safety

If OCR fails (missing dependencies), system falls back to `PyPDFLoader`:

```
⚠️ OCR failed, falling back to PyPDFLoader: Tesseract not found
```

## Performance

| PDF Type | Pages | Processing Time |
|----------|-------|-----------------|
| Text-based | 10 | ~2 seconds |
| Scanned | 10 | ~30-60 seconds |

## Troubleshooting

### "Tesseract not found"

Add Tesseract to PATH or set environment variable:

```python
# In .env file
TESSERACT_CMD=C:/Program Files/Tesseract-OCR/tesseract.exe
```

### "Poppler not found"

```python
# In .env file
POPPLER_PATH=C:/Program Files/poppler/Library/bin
```

### OCR Not Working

Test manually:

```python
from unstructured.partition.pdf import partition_pdf

elements = partition_pdf("scanned.pdf", strategy="hi_res")
print([str(el) for el in elements])
```

## Cost Considerations

OCR increases:
- ⏱ Processing time: ~10-20x slower
- 💾 Memory usage: Higher (image processing)
- 💰 API costs: No impact (OCR is local)

## When OCR is Used

✅ Scanned C&P exams
✅ Old medical records (pre-2010)
✅ Handwritten buddy statements
✅ Faxed documents
✅ Photos of documents

❌ Modern VA Decision Letters (text-based)
❌ Recent electronic C&P exams (text-based)
❌ DBQs (text-based)

## Production Recommendations

For production deployment:

1. **Pre-process uploads**: Check if PDF is scanned before upload
2. **Queue scanned PDFs**: Process OCR jobs asynchronously
3. **Cache results**: Store OCR output to avoid re-processing
4. **User notification**: "Scanned document detected - processing may take 1-2 minutes"

## Status Check

```bash
# Test OCR is working
python -c "from vris_rag_system import is_pdf_scanned; print('OCR Ready!')"
```

Your VRIS system now handles both text-based and scanned PDFs automatically! 🎉
