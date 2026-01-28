"""
VRIS™ Example Usage Scenarios
Demonstrates different workflows for the Veteran Rating Intelligence System
"""

from vris_rag_system import VRISRAGSystem
import os


def example_free_snapshot():
    """
    Example: Free Rating Snapshot
    For veterans who want a quick assessment before paying for full analysis
    """
    print("\n" + "="*80)
    print("EXAMPLE 1: FREE RATING SNAPSHOT")
    print("="*80)
    print("\nScenario: Veteran wants to see if VRIS can help before paying")
    print("Requirements: Minimum - VA Decision Letter + basic questionnaire")
    print("            Optional - Any additional medical documentation\n")
    
    vris = VRISRAGSystem(
        system_docs_folder="./system-doc",
        veteran_docs_folder="./user_pdfs/1",
        persist_directory="./chroma_db",
        model_name="gpt-4"
    )
    
    vris.initialize(force_reload=False)
    
    # Generate free snapshot
    snapshot = vris.generate_free_snapshot()
    
    print("\n" + "📊 " + "="*76)
    print("FREE SNAPSHOT RESULTS")
    print("="*78)
    print("\nEXTRACTED DATA:")
    print(snapshot['extraction'])
    print("\n\nHIGH-LEVEL ANALYSIS:")
    print(snapshot['high_level_analysis'])
    print("\n\n⚠️  " + snapshot['note'])
    print("="*78 + "\n")
    
    return snapshot


def example_initial_vre():
    """
    Example: Initial VRE for First-Time Filer
    For veterans who haven't filed yet and want to optimize their initial claim
    """
    print("\n" + "="*80)
    print("EXAMPLE 2: INITIAL VRE (FIRST-TIME FILER)")
    print("="*80)
    print("\nScenario: Veteran separating from service, never filed VA disability")
    print("Goal: Identify ALL service-connected conditions before filing")
    print("Requirements: Service Treatment Records (STRs), medical records, DD-214\n")
    
    vris = VRISRAGSystem(
        system_docs_folder="./system-doc",
        veteran_docs_folder="./user_pdfs/1",
        persist_directory="./chroma_db",
        model_name="gpt-4"
    )
    
    vris.initialize(force_reload=False)
    
    # Generate Initial VRE
    initial_vre = vris.generate_initial_vre()
    
    print("\n" + "📋 " + "="*76)
    print("INITIAL VRE RESULTS")
    print("="*78)
    print("\n🔍 VRIS-A EXTRACTION (What's in the documents):")
    print("-"*78)
    print(initial_vre['vris_a_result'])
    
    print("\n\n🧠 VRIS-B REASONING (Rating analysis & CFR mapping):")
    print("-"*78)
    print(initial_vre['vris_b_result'])
    print("="*78 + "\n")
    
    return initial_vre


def example_second_look_vre():
    """
    Example: Second Look VRE for Already-Rated Veteran
    For veterans with existing rating who suspect they're underrated
    """
    print("\n" + "="*80)
    print("EXAMPLE 3: SECOND LOOK VRE (ALREADY-RATED VETERAN)")
    print("="*80)
    print("\nScenario: Veteran has 60% rating but conditions have worsened")
    print("Goal: Identify underratings, missed conditions, secondary conditions")
    print("Requirements: VA Decision Letter, C&P exams, current medical records\n")
    
    vris = VRISRAGSystem(
        system_docs_folder="./system-doc",
        veteran_docs_folder="./user_pdfs/1",
        persist_directory="./chroma_db",
        model_name="gpt-4"
    )
    
    vris.initialize(force_reload=False)
    
    # Generate Second Look VRE
    second_look = vris.generate_second_look_vre()
    
    print("\n" + "🔎 " + "="*76)
    print("SECOND LOOK VRE RESULTS")
    print("="*78)
    print("\n🔍 VRIS-A EXTRACTION (Current rating vs. evidence):")
    print("-"*78)
    print(second_look['vris_a_result'])
    
    print("\n\n🧠 VRIS-B REASONING (Underratings & opportunities):")
    print("-"*78)
    print(second_look['vris_b_result'])
    print("="*78 + "\n")
    
    return second_look


def example_custom_extraction():
    """
    Example: Custom VRIS-A Extraction Query
    For specific data extraction needs
    """
    print("\n" + "="*80)
    print("EXAMPLE 4: CUSTOM VRIS-A EXTRACTION")
    print("="*80)
    print("\nScenario: Need to extract specific information from veteran documents")
    print("Use Case: Extract all sleep-related symptoms for sleep apnea claim\n")
    
    vris = VRISRAGSystem(
        system_docs_folder="./system-doc",
        veteran_docs_folder="./user_pdfs/1",
        persist_directory="./chroma_db",
        model_name="gpt-4"
    )
    
    vris.initialize(force_reload=False)
    
    # Custom extraction query
    query = """
    Extract all information related to sleep disorders:
    1. Any diagnoses of sleep apnea, insomnia, or sleep disturbances
    2. Sleep study results and findings
    3. Symptoms: snoring, gasping, fatigue, daytime sleepiness
    4. CPAP usage or other treatments
    5. Impact on daily functioning and work
    6. Any conditions that may cause or worsen sleep issues (PTSD, chronic pain, etc.)
    """
    
    result = vris.vris_a_extract(query)
    
    print("\n" + "📄 EXTRACTION RESULTS:")
    print("="*78)
    print(result)
    print("="*78 + "\n")
    
    return result


def example_custom_reasoning():
    """
    Example: Custom VRIS-B Reasoning Query
    For specific analysis questions
    """
    print("\n" + "="*80)
    print("EXAMPLE 5: CUSTOM VRIS-B REASONING")
    print("="*80)
    print("\nScenario: Specific question about rating criteria")
    print("Use Case: Analyze if knee condition qualifies for higher rating\n")
    
    vris = VRISRAGSystem(
        system_docs_folder="./system-doc",
        veteran_docs_folder="./user_pdfs/1",
        persist_directory="./chroma_db",
        model_name="gpt-4"
    )
    
    vris.initialize(force_reload=False)
    
    # Custom reasoning query
    query = """
    Based on the veteran's knee condition evidence:
    1. What is the current VA rating for the knee condition?
    2. What are the CFR criteria for knee ratings (38 CFR 4.71a)?
    3. Does the evidence support a higher rating? At what percentage?
    4. What specific measurements or findings would support the increase?
    5. Are there any secondary conditions (e.g., hip, back, gait issues)?
    
    Provide detailed CFR citations and evidence mappings.
    """
    
    result = vris.vris_b_analyze(query)
    
    print("\n" + "🧠 REASONING RESULTS:")
    print("="*78)
    print(result)
    print("="*78 + "\n")
    
    return result


def example_dual_pipeline_with_confidence():
    """
    Example: Dual Pipeline with Agreement Analysis
    Running both VRIS-A and VRIS-B, checking for 90%+ agreement
    """
    print("\n" + "="*80)
    print("EXAMPLE 6: DUAL PIPELINE WITH AGREEMENT SCORING")
    print("="*80)
    print("\nScenario: Full VRIS analysis with cross-validation")
    print("Goal: Only report findings where VRIS-A and VRIS-B agree by 90%+\n")
    
    vris = VRISRAGSystem(
        system_docs_folder="./system-doc",
        veteran_docs_folder="./user_pdfs/1",
        persist_directory="./chroma_db",
        model_name="gpt-4"
    )
    
    vris.initialize(force_reload=False)
    
    # Extraction query for VRIS-A
    extraction_query = """
    Extract all rated conditions with:
    - Current rating percentage
    - Diagnostic Code
    - Evidence of severity (symptoms, ROM, test results)
    - Any functional limitations documented
    """
    
    # Reasoning query for VRIS-B
    reasoning_query = """
    For each rated condition:
    1. Map current rating to CFR criteria
    2. Compare evidence to rating requirements
    3. Identify if evidence supports higher rating
    4. Provide confidence score (0-100%)
    5. Cite specific CFR sections
    """
    
    # Run dual pipeline
    result = vris.dual_pipeline_analysis(extraction_query, reasoning_query)
    
    print("\n" + "🔄 DUAL PIPELINE RESULTS:")
    print("="*78)
    print("\n📊 VRIS-A EXTRACTION:")
    print("-"*78)
    print(result['vris_a_extraction'])
    
    print("\n\n🔬 VRIS-B REASONING:")
    print("-"*78)
    print(result['vris_b_reasoning'])
    
    print("\n\n⚖️  AGREEMENT ANALYSIS:")
    print("-"*78)
    if result['agreement_score']:
        print(f"Agreement Score: {result['agreement_score']}%")
        if result['agreement_score'] >= 90:
            print("✅ High confidence - VRIS-A and VRIS-B strongly agree")
        else:
            print("⚠️  Lower confidence - Requires additional review")
    else:
        print("Note: Agreement scoring requires additional implementation")
    print("="*78 + "\n")
    
    return result


def example_document_classification():
    """
    Example: Document Classification
    Shows how VRIS classifies uploaded veteran documents
    """
    print("\n" + "="*80)
    print("EXAMPLE 7: DOCUMENT CLASSIFICATION")
    print("="*80)
    print("\nScenario: Veteran uploads multiple PDFs")
    print("Goal: Automatically classify each document type for targeted processing\n")
    
    vris = VRISRAGSystem(
        system_docs_folder="./system-doc",
        veteran_docs_folder="./user_pdfs/1",
        persist_directory="./chroma_db",
        model_name="gpt-4"
    )
    
    # Just initialize to see classification
    vris.initialize(force_reload=False)
    
    print("\n📁 CLASSIFIED DOCUMENTS:")
    print("="*78)
    for doc_type, docs in vris.classified_docs.items():
        if docs:
            doc_name = vris.classifier.DOCUMENT_TYPES.get(doc_type, doc_type)
            print(f"\n{doc_name}:")
            print(f"  └─ {len(docs)} pages/documents")
            # Show first few filenames
            filenames = set([doc.metadata.get('filename', 'Unknown') for doc in docs])
            for filename in list(filenames)[:3]:
                print(f"     • {filename}")
    print("="*78 + "\n")


def run_all_examples():
    """Run all example scenarios"""
    print("\n" + "🎯 "*40)
    print("VRIS™ COMPREHENSIVE EXAMPLE SUITE")
    print("Veteran Rating Intelligence System - All Workflows")
    print("🎯 "*40 + "\n")
    
    print("This suite demonstrates all VRIS capabilities:")
    print("  1. Free Rating Snapshot")
    print("  2. Initial VRE (First-Time Filer)")
    print("  3. Second Look VRE (Already-Rated)")
    print("  4. Custom VRIS-A Extraction")
    print("  5. Custom VRIS-B Reasoning")
    print("  6. Dual Pipeline with Agreement")
    print("  7. Document Classification")
    
    input("\nPress Enter to continue...")
    
    # Run examples (comment out as needed)
    try:
        example_document_classification()
        input("\nPress Enter for next example...")
        
        example_free_snapshot()
        input("\nPress Enter for next example...")
        
        example_custom_extraction()
        input("\nPress Enter for next example...")
        
        example_custom_reasoning()
        input("\nPress Enter for next example...")
        
        example_dual_pipeline_with_confidence()
        input("\nPress Enter for next example...")
        
        example_initial_vre()
        input("\nPress Enter for next example...")
        
        example_second_look_vre()
        
        print("\n" + "✅ "*40)
        print("ALL EXAMPLES COMPLETED SUCCESSFULLY")
        print("✅ "*40 + "\n")
        
    except Exception as e:
        print(f"\n❌ Error running examples: {e}")
        print("Make sure you have:")
        print("  1. OPENAI_API_KEY set in .env file")
        print("  2. Documents in ./system-doc folder")
        print("  3. Veteran documents in ./user_pdfs/1 folder")


if __name__ == "__main__":
    # Run specific example or all
    import sys
    
    if len(sys.argv) > 1:
        example_num = sys.argv[1]
        examples = {
            '1': example_free_snapshot,
            '2': example_initial_vre,
            '3': example_second_look_vre,
            '4': example_custom_extraction,
            '5': example_custom_reasoning,
            '6': example_dual_pipeline_with_confidence,
            '7': example_document_classification
        }
        
        if example_num in examples:
            examples[example_num]()
        else:
            print("Invalid example number. Use 1-7 or run without arguments for all.")
    else:
        run_all_examples()
