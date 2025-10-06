"""
Preload Documents Script
Run this ONCE to load your college documents into the system
Then students can start asking questions immediately
"""

import os
import glob
from pathlib import Path
from ragpipeline import DocumentIngestionPipeline, initialize_vector_store

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
    
    # Initialize ingestion pipeline
    print("🔄 Initializing document ingestion pipeline...")
    ingestion_pipeline = DocumentIngestionPipeline(vector_store)
    print("✅ Pipeline ready")
    
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
    
    # Find all PDF and CSV files
    pdf_files = glob.glob(os.path.join(data_directory, "*.pdf"))
    csv_files = glob.glob(os.path.join(data_directory, "*.csv"))
    
    all_files = pdf_files + csv_files
    
    if not all_files:
        print(f"\n⚠️  No PDF or CSV files found in '{data_directory}/'")
        print("\n💡 Add your college documents and run this script again.")
        return
    
    print(f"\n📂 Found {len(all_files)} files:")
    print(f"   - PDFs: {len(pdf_files)}")
    print(f"   - CSVs: {len(csv_files)}")
    print()
    
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