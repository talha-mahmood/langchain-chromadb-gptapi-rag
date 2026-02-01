# Quick Start Guide with Sample Documents

## What I've Done

1. ✅ **Added DOCX support** - The system now loads both PDF and DOCX files
2. ✅ **Installed docx2txt** - Required library for reading Word documents
3. ✅ **Created sample veteran documents** for testing:

### Sample Documents Created (in `user_pdfs/1/`):

1. **sample_va_decision_letter.txt**
   - Current 60% combined rating
   - Three service-connected conditions:
     - Degenerative Disc Disease (Back): 20%
     - Bilateral Knee Strain: 10% each (20% combined)
     - PTSD: 30%

2. **sample_cp_exam_back.txt**
   - Detailed C&P exam for back condition
   - Shows evidence of radiculopathy (nerve damage)
   - Examiner suggests considering separate rating for nerve condition
   - **VRIS Opportunity**: Could identify underrating

3. **sample_cp_exam_ptsd.txt**
   - Detailed mental health evaluation
   - Currently rated 30% but examiner notes symptoms meet 50% criteria
   - Documents work impairment, relationship strain, sleep issues
   - **VRIS Opportunity**: Clear underrating identified

4. **sample_private_sleep_study.txt**
   - Sleep apnea diagnosis (moderate severity, AHI 22)
   - Requires CPAP nightly (qualifies for 50% rating if service-connected)
   - Doctor provides nexus opinion linking to PTSD and chronic pain
   - **VRIS Opportunity**: Missed secondary condition

## Expected VRIS Findings

When you run VRIS on these sample documents, it should identify:

### Underrated Conditions:
1. **PTSD**: Currently 30% → Should be 50%
   - Evidence: C&P examiner specifically notes 50% criteria met
   - Impact: Work absences, decreased productivity, relationship issues

### Missed Conditions:
1. **Sleep Apnea**: Not currently rated → Should be 50%
   - Evidence: Sleep study shows moderate OSA requiring CPAP
   - Service connection: Secondary to PTSD and chronic pain
   - DC 6847: 50% for breathing assistance device (CPAP)

2. **Radiculopathy (Left Leg)**: Not separately rated → Should be 10-20%
   - Evidence: C&P exam shows nerve damage (positive SLR, decreased sensation)
   - DC 8520: Incomplete paralysis of sciatic nerve
   - Currently bundled into back rating

3. **Insomnia**: Not rated → Should be 0-10%
   - Evidence: Sleep study documents chronic insomnia
   - Secondary to PTSD (nightmares, hypervigilance)

### Potential Rating Increase:
- **Current**: 60% combined
- **Potential**: 80-90% combined
- **Monthly increase**: ~$1,000-1,500
- **Annual increase**: ~$12,000-18,000

## How to Run Now

```bash
python vris_rag_system.py
```

Then select:
- **Option 3**: Second Look VRE (best for this scenario - already-rated veteran)

### Good Queries to Try:

**VRIS-A Extraction:**
```
Extract all service-connected conditions with their current ratings, 
diagnostic codes, and compare the evidence in the C&P exams to what 
was actually rated. Identify any conditions mentioned in medical 
records that are not currently service-connected.
```

**VRIS-B Reasoning:**
```
Analyze the veteran's current 60% rating against the evidence in the 
C&P exams and private medical records. Identify conditions that appear 
underrated, conditions that were missed entirely, and any secondary 
conditions that should be claimed. Provide specific CFR citations and 
confidence scores.
```

## What VRIS Should Detect

✅ **PTSD Underrating**: Should recommend increase from 30% to 50% based on C&P exam findings

✅ **Sleep Apnea Opportunity**: Should identify as strong secondary condition with nexus letter

✅ **Radiculopathy**: Should recognize nerve damage documented but not separately rated

✅ **Total Rating Impact**: Should calculate potential increase to 80-90% combined

## Next Steps

1. Run `python vris_rag_system.py`
2. Choose workflow option 3 (Second Look VRE)
3. Review the VRIS-A extraction and VRIS-B reasoning outputs
4. Test custom queries if needed

Your system-doc folder already has the VRIS documentation (in DOCX format), and now you have realistic veteran documents to test against!
