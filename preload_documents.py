"""
Preload Documents Script
Run this ONCE to load your college documents into the system
Then students can start asking questions immediately
"""

import os
import glob
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List

import pandas as pd
from langchain.document_loaders import PyPDFLoader
from langchain.schema import Document

from ragpipeline import DocumentIngestionPipeline, initialize_vector_store


@dataclass
class PreloadedContent:
    """In-memory snapshot of CSV tables and PDF pages."""

    csv_data: Dict[str, pd.DataFrame]
    pdf_pages: Dict[str, List[Document]]
    csv_files: List[str]
    pdf_files: List[str]


_DOCUMENT_CACHE: Dict[str, object] = {"directory": None, "content": None}


def load_document_cache(data_directory: str = "college_data",
                        force_reload: bool = False) -> PreloadedContent:
    """Eagerly load all CSV/PDF documents into memory once."""

    global _DOCUMENT_CACHE

    directory = Path(data_directory).resolve()

    # Directory missing – cache as empty snapshot
    if not directory.exists():
        content = PreloadedContent(csv_data={},
                                   pdf_pages={},
                                   csv_files=[],
                                   pdf_files=[])
        _DOCUMENT_CACHE = {"directory": directory, "content": content}
        return content

    cached_dir = _DOCUMENT_CACHE.get("directory")
    cached_content = _DOCUMENT_CACHE.get("content")

    if (not force_reload and cached_dir == directory
            and isinstance(cached_content, PreloadedContent)):
        return cached_content

    csv_data: Dict[str, pd.DataFrame] = {}
    pdf_pages: Dict[str, List[Document]] = {}
    csv_files: List[str] = []
    pdf_files: List[str] = []

    for csv_path in sorted(directory.glob("*.csv")):
        csv_files.append(str(csv_path))
        csv_data[csv_path.name] = pd.read_csv(csv_path)

    for pdf_path in sorted(directory.glob("*.pdf")):
        pdf_files.append(str(pdf_path))
        loader = PyPDFLoader(str(pdf_path))
        pdf_pages[pdf_path.name] = loader.load()

    content = PreloadedContent(csv_data=csv_data,
                               pdf_pages=pdf_pages,
                               csv_files=csv_files,
                               pdf_files=pdf_files)

    _DOCUMENT_CACHE = {"directory": directory, "content": content}
    return content


def clear_document_cache():
    """Reset the in-memory cache (handy for tests or reloads)."""

    global _DOCUMENT_CACHE
    _DOCUMENT_CACHE = {"directory": None, "content": None}

def preload_documents(data_directory: str = "college_data"):
    """
    Load all PDF and CSV files from a directory into the vector database
    """
    
    print("="*60)
    print("📚 COLLEGE CHATBOT - Document Preloader")
    print("="*60)
    
    # Initialize vector store
    print("\n🔄 Initializing vector store...")
    vector_store = initialize_vector_store()
    print("✅ Vector store initialized")
    
    # Check if directory exists
    if not os.path.exists(data_directory):
        print(f"\n❌ Directory '{data_directory}' not found!")
        print(f"📁 Creating directory: {data_directory}")
        os.makedirs(data_directory)
        print(f"\n💡 Please place your college documents in '{data_directory}/' directory:")
        print("   - PDFs: prospectus, syllabus, admission guidelines")
        print("   - CSVs: fees, courses, eligibility criteria")
        print("\nThen run this script again.")
        return

    # Load documents once into memory
    print("\n📥 Loading documents into memory (one-time)...")
    content = load_document_cache(data_directory)

    total_files = len(content.pdf_files) + len(content.csv_files)
    if total_files == 0:
        print(f"\n⚠️  No PDF or CSV files found in '{data_directory}/'")
        print("\n💡 Add your college documents and run this script again.")
        return

    print(f"✅ Cached {total_files} files:")
    print(f"   - PDFs: {len(content.pdf_files)}")
    print(f"   - CSVs: {len(content.csv_files)}")

    # Initialize ingestion pipeline with cached data
    print("\n🔄 Initializing document ingestion pipeline...")
    ingestion_pipeline = DocumentIngestionPipeline(
        vector_store,
        preloaded_csvs=content.csv_data,
        preloaded_pdfs=content.pdf_pages)
    print("✅ Pipeline ready (using in-memory data)")

    # Merge in the same order: PDFs first for consistency with previous runs
    all_files = content.pdf_files + content.csv_files
    
    # Process each file
    total_chunks = 0
    successful_files = []
    failed_files = []
    
    for idx, file_path in enumerate(all_files, 1):
        filename = os.path.basename(file_path)
        print(f"[{idx}/{len(all_files)}] Processing: {filename}")
        
        try:
            chunks = ingestion_pipeline.ingest_file(file_path, filename)
            total_chunks += len(chunks)
            successful_files.append(filename)
            print(f"    ✅ Created {len(chunks)} chunks")
        except Exception as e:
            failed_files.append(filename)
            print(f"    ❌ Failed: {str(e)}")
    
    # Summary
    print("\n" + "="*60)
    print("📊 PRELOAD SUMMARY")
    print("="*60)
    print(f"✅ Successfully loaded: {len(successful_files)} files")
    print(f"❌ Failed: {len(failed_files)} files")
    print(f"📦 Total chunks created: {total_chunks}")
    
    if successful_files:
        print("\n📚 Loaded files:")
        for f in successful_files:
            print(f"   - {f}")
    
    if failed_files:
        print("\n⚠️  Failed files:")
        for f in failed_files:
            print(f"   - {f}")
    
    print("\n" + "="*60)
    if total_chunks > 0:
        print("✅ SUCCESS! Documents loaded into knowledge base.")
        print("\n🚀 Next steps:")
        print("   1. Start backend: python main.py")
        print("   2. Start student UI: python app_gradio_student.py")
        print("   3. Students can now ask questions!")
    else:
        print("❌ No documents were loaded. Please check errors above.")
    print("="*60)

def create_sample_data():
    """Create sample college data if none exists"""
    
    data_dir = "college_data"
    os.makedirs(data_dir, exist_ok=True)
    
    # Create sample fees.csv
    fees_csv = os.path.join(data_dir, "fees.csv")
    if not os.path.exists(fees_csv):
        with open(fees_csv, 'w') as f:
            f.write("""Branch,Specialization,Annual Fee,Total Fee (4 Years),Year
CSE,Cybersecurity,130000,520000,2025
CSE,AI & ML,135000,540000,2025
CSE,Data Science,135000,540000,2025
CSE,IoT,130000,520000,2025
ECE,VLSI,120000,480000,2025
ECE,Embedded Systems,120000,480000,2025
Mechanical,Robotics,115000,460000,2025
Mechanical,General,110000,440000,2025
Civil,General,105000,420000,2025
Civil,Structural Engineering,108000,432000,2025""")
        print(f"✅ Created sample: {fees_csv}")
    
    # Create sample courses.csv
    courses_csv = os.path.join(data_dir, "courses.csv")
    if not os.path.exists(courses_csv):
        with open(courses_csv, 'w') as f:
            f.write("""Branch,Duration,Seats,Eligibility,Accreditation
CSE,4 years,120,60% in 12th with Physics Chemistry Maths,NAAC A+
ECE,4 years,90,60% in 12th with Physics Chemistry Maths,NAAC A+
Mechanical,4 years,60,60% in 12th with Physics Chemistry Maths,NBA
Civil,4 years,60,60% in 12th with Physics Chemistry Maths,NBA
AI & ML,4 years,60,60% in 12th with Physics Chemistry Maths,NAAC A+""")
        print(f"✅ Created sample: {courses_csv}")
    
    # Create sample eligibility.csv
    eligibility_csv = os.path.join(data_dir, "eligibility.csv")
    if not os.path.exists(eligibility_csv):
        with open(eligibility_csv, 'w') as f:
            f.write("""Program,Entry Type,Minimum Marks,Required Subjects,Additional Requirements
BTech Regular,Direct Entry,60%,Physics Chemistry Maths,State entrance exam or JEE Main
BTech Lateral,Lateral Entry,50%,Diploma in relevant branch,3 years diploma with 50% aggregate
MTech,Postgraduate,60% in BTech,Relevant UG degree,GATE score preferred but not mandatory""")
        print(f"✅ Created sample: {eligibility_csv}")

if __name__ == "__main__":
    import sys
    
    print("\n🎓 College Chatbot - Document Preloader\n")
    
    # Check if user wants to create sample data
    if len(sys.argv) > 1 and sys.argv[1] == "--create-sample":
        print("📝 Creating sample college data...")
        create_sample_data()
        print("\n✅ Sample data created in 'college_data/' directory")
        print("📚 You can now edit these files with your actual college data\n")
        
        response = input("Do you want to load these sample files now? (y/n): ")
        if response.lower() == 'y':
            preload_documents()
    else:
        # Normal preload
        data_dir = "college_data"
        
        if not os.path.exists(data_dir) or not glob.glob(os.path.join(data_dir, "*.*")):
            print(f"⚠️  No '{data_dir}/' directory or files found.\n")
            response = input("Do you want to create sample data? (y/n): ")
            if response.lower() == 'y':
                create_sample_data()
                print()
                preload_documents()
            else:
                print(f"\n💡 Please create '{data_dir}/' directory and add your files:")
                print("   - PDFs: prospectus, syllabus")
                print("   - CSVs: fees, courses, eligibility")
                print(f"\nThen run: python {sys.argv[0]}")
        else:
            preload_documents()
    
    print()