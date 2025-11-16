"""
Create All College Data CSV Files - FIXED VERSION
Run this script to generate all 10 sample CSV files
"""

import os

def create_college_data_files():
    """Create all college data CSV files"""
    
    # Create directory
    os.makedirs("college_data", exist_ok=True)
    
    print("📚 Creating College Data Files...")
    print("="*60)
    
    # 1. FEES.CSV
    fees_csv = """Branch,Specialization,Annual Fee,Semester Fee,Total Fee (4 Years),Caution Deposit,Other Fees,Year
CSE,Cybersecurity,130000,65000,520000,10000,5000,2025
CSE,AI & Machine Learning,135000,67500,540000,10000,5000,2025
CSE,Data Science,135000,67500,540000,10000,5000,2025
CSE,IoT,130000,65000,520000,10000,5000,2025
CSE,Blockchain Technology,130000,65000,520000,10000,5000,2025
CSE,Cloud Computing,128000,64000,512000,10000,5000,2025
CSE,General,125000,62500,500000,10000,5000,2025
ECE,VLSI Design,120000,60000,480000,10000,5000,2025
ECE,Embedded Systems,120000,60000,480000,10000,5000,2025
ECE,Communication Systems,118000,59000,472000,10000,5000,2025
ECE,Signal Processing,118000,59000,472000,10000,5000,2025
ECE,General,115000,57500,460000,10000,5000,2025
Mechanical,Robotics & Automation,115000,57500,460000,10000,5000,2025
Mechanical,Automotive Engineering,113000,56500,452000,10000,5000,2025
Mechanical,Manufacturing,110000,55000,440000,10000,5000,2025
Mechanical,Thermal Engineering,110000,55000,440000,10000,5000,2025
Mechanical,General,108000,54000,432000,10000,5000,2025
Civil,Structural Engineering,108000,54000,432000,10000,5000,2025
Civil,Environmental Engineering,106000,53000,424000,10000,5000,2025
Civil,Transportation Engineering,105000,52500,420000,10000,5000,2025
Civil,Construction Management,105000,52500,420000,10000,5000,2025
Civil,General,103000,51500,412000,10000,5000,2025
EEE,Power Systems,112000,56000,448000,10000,5000,2025
EEE,Control Systems,112000,56000,448000,10000,5000,2025
EEE,Renewable Energy,115000,57500,460000,10000,5000,2025
EEE,General,110000,55000,440000,10000,5000,2025
IT,Software Engineering,130000,65000,520000,10000,5000,2025
IT,Network Security,128000,64000,512000,10000,5000,2025
IT,General,125000,62500,500000,10000,5000,2025
AIDS,Artificial Intelligence,140000,70000,560000,10000,5000,2025
AIDS,Data Science,140000,70000,560000,10000,5000,2025"""
    
    with open("college_data/fees.csv", "w", encoding='utf-8') as f:
        f.write(fees_csv)
    print("✅ Created: fees.csv (31 rows)")
    
    # 2. COURSES.CSV
    courses_csv = """Branch,Full Name,Duration,Total Seats,Regular Seats,Lateral Entry Seats,Eligibility,Accreditation,Placement Rate,Average Package,Highest Package
CSE,Computer Science and Engineering,4 years,180,120,60,60% in 12th with Physics Chemistry Maths,NAAC A+ NBA,95%,8.5 LPA,45 LPA
ECE,Electronics and Communication Engineering,4 years,120,90,30,60% in 12th with Physics Chemistry Maths,NAAC A+ NBA,92%,7.2 LPA,35 LPA
Mechanical,Mechanical Engineering,4 years,120,90,30,60% in 12th with Physics Chemistry Maths,NBA,88%,6.5 LPA,28 LPA
Civil,Civil Engineering,4 years,90,60,30,60% in 12th with Physics Chemistry Maths,NBA,85%,6.0 LPA,22 LPA
EEE,Electrical and Electronics Engineering,4 years,90,60,30,60% in 12th with Physics Chemistry Maths,NBA,90%,7.0 LPA,30 LPA
IT,Information Technology,4 years,120,90,30,60% in 12th with Physics Chemistry Maths,NAAC A+,94%,8.2 LPA,42 LPA
AIDS,Artificial Intelligence and Data Science,4 years,90,60,30,60% in 12th with Physics Chemistry Maths,NAAC A+,96%,9.5 LPA,50 LPA"""
    
    with open("college_data/courses.csv", "w", encoding='utf-8') as f:
        f.write(courses_csv)
    print("✅ Created: courses.csv (7 rows)")
    
    # 3. ELIGIBILITY.CSV
    eligibility_csv = """Program,Entry Type,Minimum Marks,Required Subjects,Entrance Exam,Additional Requirements,Age Limit,Category Relaxation
BTech,Direct Entry,60%,Physics Chemistry Maths,State Entrance or JEE Main,Valid scorecard,No limit,5% relaxation for SC/ST
BTech,Lateral Entry,50%,Diploma in relevant branch,ECET or Institute Test,3 years diploma with minimum 50%,No limit,5% relaxation for SC/ST
BTech,Management Quota,55%,Physics Chemistry Maths,Not required,Direct admission through college,No limit,No relaxation
MTech,Postgraduate Entry,60% in BTech,Relevant UG degree,GATE,Valid GATE score preferred,30 years,5% relaxation for SC/ST
MTech,Sponsored Category,55% in BTech,Relevant UG degree,Not required,Sponsorship letter from employer,35 years,Not applicable
MBA,Postgraduate Entry,50% in graduation,Any degree,CAT MAT CMAT,Valid scorecard required,28 years,5% relaxation for SC/ST
MCA,Postgraduate Entry,50% in graduation,Any degree with Maths,NIMCET,Maths in 10+2 or graduation,30 years,5% relaxation for SC/ST
PhD,Research,60% in PG,Relevant PG degree,Entrance Test Interview,Research proposal required,40 years,5% relaxation for SC/ST"""
    
    with open("college_data/eligibility.csv", "w", encoding='utf-8') as f:
        f.write(eligibility_csv)
    print("✅ Created: eligibility.csv (8 rows)")
    
    # 4. SPECIALIZATIONS.CSV
    specializations_csv = """Branch,Specialization,Description,Career Opportunities,Industry Demand,Starting From Year,Key Subjects
CSE,Cybersecurity,Focus on network security ethical hacking and cyber defense,Security Analyst Penetration Tester SOC Analyst,Very High,2nd Year,Cryptography Network Security Ethical Hacking
CSE,AI & Machine Learning,Study of artificial intelligence machine learning and deep learning,ML Engineer AI Researcher Data Scientist,Very High,2nd Year,Machine Learning Deep Learning NLP Computer Vision
CSE,Data Science,Analysis of big data statistical modeling and business intelligence,Data Analyst Business Analyst Data Engineer,Very High,2nd Year,Big Data Analytics Statistics Python R Programming
CSE,IoT,Internet of Things embedded systems and smart devices,IoT Developer Embedded Engineer Smart City Expert,High,2nd Year,Sensor Networks Embedded Systems IoT Protocols
CSE,Blockchain Technology,Distributed ledger technology cryptocurrency and smart contracts,Blockchain Developer Crypto Analyst Smart Contract Developer,High,3rd Year,Blockchain Fundamentals Cryptocurrency Smart Contracts
CSE,Cloud Computing,Cloud infrastructure services and deployment models,Cloud Architect DevOps Engineer Cloud Consultant,Very High,2nd Year,AWS Azure Cloud Architecture Microservices
ECE,VLSI Design,Very Large Scale Integration circuit design and fabrication,VLSI Engineer Chip Designer Verification Engineer,High,2nd Year,Digital Design CMOS VLSI System Verilog
ECE,Embedded Systems,Microcontroller programming and embedded software,Embedded Developer Firmware Engineer IoT Engineer,High,2nd Year,Microcontrollers ARM RTOS Embedded C
ECE,Communication Systems,Wireless communication optical fiber and satellite systems,Telecom Engineer RF Engineer Network Engineer,Medium,2nd Year,Digital Communication Wireless Networks Antenna Theory
Mechanical,Robotics & Automation,Industrial robots automated systems and control,Robotics Engineer Automation Engineer Control Systems Engineer,High,2nd Year,Robotics Industrial Automation PLC Programming
Mechanical,Automotive Engineering,Vehicle design manufacturing and electric vehicles,Automobile Engineer EV Designer Automotive Analyst,Very High,2nd Year,Automobile Engineering EV Technology Vehicle Dynamics
Civil,Structural Engineering,Building design structural analysis and earthquake engineering,Structural Engineer Design Consultant Site Engineer,High,2nd Year,Structural Analysis Design of Structures Earthquake Engineering
EEE,Power Systems,Electrical grids power generation and distribution,Power Engineer Grid Engineer Electrical Consultant,High,2nd Year,Power Systems Power Electronics Grid Integration
EEE,Renewable Energy,Solar wind and green energy technologies,Renewable Energy Engineer Solar Consultant Green Energy Analyst,Very High,2nd Year,Solar Energy Wind Energy Energy Storage
IT,Software Engineering,Software development lifecycle testing and project management,Software Engineer Developer Tech Lead,Very High,2nd Year,Software Engineering Agile Testing DevOps"""
    
    with open("college_data/specializations.csv", "w", encoding='utf-8') as f:
        f.write(specializations_csv)
    print("✅ Created: specializations.csv (15 rows)")
    
    # 5. ADMISSION_PROCESS.CSV
    admission_csv = """Entry Type,Step Number,Step Name,Description,Timeline,Documents Required,Fees,Contact
Direct Entry,1,Entrance Exam,Appear for State Entrance or JEE Main exam,April to May,Hall Ticket ID Proof,As per exam body,examhelpdesk@college.edu
Direct Entry,2,Counselling Registration,Register for state counselling with rank card,June,Rank Card Certificates ID Proof Caste Certificate,Rs 1000,counselling@college.edu
Direct Entry,3,Choice Filling,Fill college and branch preferences online,June to July,Login Credentials,Free,counselling@college.edu
Direct Entry,4,Seat Allotment,Check allotment results and download letter,July,None,Free,admissions@college.edu
Direct Entry,5,Document Verification,Visit college with original documents,July to August,10th 12th TC Rank Card Aadhar,Free,admissions@college.edu
Direct Entry,6,Fee Payment,Pay first semester fee and confirm admission,August,Allotment Letter Bank Details,As per fee structure,accounts@college.edu
Direct Entry,7,Orientation,Attend orientation program and get ID card,August,Admission Receipt Photo,Free,student.affairs@college.edu
Lateral Entry,1,ECET Exam,Appear for ECET or Institute entrance test,May,Hall Ticket ID Proof Diploma Certificate,As per exam body,lateralentry@college.edu
Lateral Entry,2,Counselling,Participate in lateral entry counselling,June to July,ECET Rank Card Diploma Marks,Rs 1000,lateralentry@college.edu
Lateral Entry,3,Seat Allotment,Receive allotment and download letter,July,None,Free,lateralentry@college.edu
Management Quota,1,Application,Submit application form with documents,June to August,Application Form 10th 12th Aadhar,Rs 500,mgmtadmissions@college.edu
Management Quota,2,Counselling,Attend counselling session at college,Within 3 days,All Original Documents,Free,mgmtadmissions@college.edu"""
    
    with open("college_data/admission_process.csv", "w", encoding='utf-8') as f:
        f.write(admission_csv)
    print("✅ Created: admission_process.csv (12 rows)")
    
    # 6. FACILITIES.CSV
    facilities_csv = """Facility Type,Facility Name,Description,Capacity,Timing,Charges,Location,Contact
Library,Central Library,4 lakh books 200+ journals digital library,500 students,8 AM to 8 PM weekdays,Free for students,Main Block Ground Floor,library@college.edu
Hostel,Boys Hostel,AC and Non-AC rooms mess and laundry,800 students,24x7 access,Rs 80000 per year AC Rs 60000 Non-AC,Campus North Block,boyshostel@college.edu
Hostel,Girls Hostel,AC and Non-AC rooms mess and security,600 students,24x7 access,Rs 80000 per year AC Rs 60000 Non-AC,Campus South Block,girlshostel@college.edu
Lab,CSE Lab,300+ computers latest software,60 students per batch,8 AM to 6 PM,Included in fee,CSE Block 2nd Floor,cselab@college.edu
Lab,ECE Lab,VLSI embedded systems communication lab,40 students per batch,8 AM to 6 PM,Included in fee,ECE Block 1st Floor,ecelab@college.edu
Sports,Indoor Sports Complex,Badminton table tennis chess carrom,100 students,6 AM to 9 PM,Free for students,Sports Complex,sports@college.edu
Sports,Gymnasium,Modern gym equipment fitness training,50 students,6 AM to 9 PM,Rs 1000 per semester,Sports Complex 1st Floor,gym@college.edu
Healthcare,Medical Center,General physician nurse emergency care,20 patients,9 AM to 6 PM weekdays,Free consultation,Admin Block Ground,medical@college.edu
Transport,College Bus,AC buses covering 30+ routes,1500 students,As per route schedule,Rs 15000 to Rs 25000 per year,Transport Office,transport@college.edu
Cafeteria,Main Canteen,Veg non-veg snacks beverages,300 seating,7 AM to 8 PM,Rs 50 to Rs 150 per meal,Central Block Ground,canteen@college.edu"""
    
    with open("college_data/facilities.csv", "w", encoding='utf-8') as f:
        f.write(facilities_csv)
    print("✅ Created: facilities.csv (10 rows)")
    
    # 7. PLACEMENTS.CSV
    placements_csv = """Academic Year,Branch,Companies Visited,Students Placed,Placement Percentage,Average Package,Median Package,Highest Package,Top Recruiters
2024,CSE,150,165,95%,8.5 LPA,7.5 LPA,45 LPA,Google Microsoft Amazon TCS Infosys
2024,ECE,120,108,92%,7.2 LPA,6.5 LPA,35 LPA,Intel Qualcomm Samsung Broadcom TCS
2024,Mechanical,80,75,88%,6.5 LPA,6.0 LPA,28 LPA,Bosch Maruti L&T Tata Motors Ashok Leyland
2024,Civil,60,50,85%,6.0 LPA,5.5 LPA,22 LPA,L&T Shapoorji Pallonji GMR Infra DLF
2024,IT,100,112,94%,8.2 LPA,7.0 LPA,42 LPA,Wipro Cognizant Accenture HCL Tech Mahindra
2024,AIDS,80,85,96%,9.5 LPA,8.5 LPA,50 LPA,Amazon Flipkart Walmart Oracle Goldman Sachs"""
    
    with open("college_data/placements.csv", "w", encoding='utf-8') as f:
        f.write(placements_csv)
    print("✅ Created: placements.csv (6 rows)")
    
    # 8. SCHOLARSHIPS.CSV
    scholarships_csv = """Scholarship Name,Type,Eligibility,Amount,Duration,Application Process,Deadline,Contact
Merit Scholarship,Academic Merit,Above 90% in 12th or entrance exam,100% tuition fee waiver,4 years,Automatic based on marks,Not applicable,scholarships@college.edu
Merit Scholarship,Academic Merit,85% to 90% in 12th or entrance exam,50% tuition fee waiver,4 years,Automatic based on marks,Not applicable,scholarships@college.edu
Sports Scholarship,Sports Achievement,State level sports participation,Rs 50000 per year,4 years,Apply with certificates,July 31st,sports@college.edu
SC/ST Scholarship,Government,SC/ST caste certificate and income below 2.5L,As per government norms,4 years,Through state portal,August 31st,scstscholarship@college.edu
Girl Child Scholarship,Gender Support,Female students with above 70%,Rs 25000 per year,4 years,Apply at admission,August 15th,girlscholarship@college.edu
Fee Reimbursement,Government,Annual income below 2 lakhs,Full tuition fee,4 years,Through state portal,August 31st,feereimbursement@college.edu"""
    
    with open("college_data/scholarships.csv", "w", encoding='utf-8') as f:
        f.write(scholarships_csv)
    print("✅ Created: scholarships.csv (6 rows)")
    
    # 9. FAQ.CSV
    faq_csv = """Category,Question,Answer
Admission,What is the admission process?,Admission is through State Entrance or JEE Main followed by counselling and seat allotment
Admission,When does admission start?,Admissions typically start in June after entrance exam results
Fees,What is the total fee for BTech?,Fees range from Rs 4.12L to Rs 5.60L for 4 years depending on branch and specialization
Fees,Are there any scholarships?,Yes merit-based government and need-based scholarships available
Courses,Which branches are available?,CSE ECE Mechanical Civil EEE IT and AI&DS with various specializations
Placements,What is the placement record?,Overall 90%+ placement with average package of 7-9 LPA
Facilities,Is hostel facility available?,Yes separate hostels for boys and girls with AC and non-AC options
Eligibility,What is the minimum percentage required?,60% in 12th with PCM for general category and 55% for reserved categories"""
    
    with open("college_data/faq.csv", "w", encoding='utf-8') as f:
        f.write(faq_csv)
    print("✅ Created: faq.csv (8 rows)")
    
    # 10. CONTACT_INFO.CSV
    contact_csv = """Department,Designation,Name,Email,Phone,Office Location
Admissions,Director Admissions,Dr. Rajesh Kumar,admissions@college.edu,+91-040-12345678,Admin Block Room 101
Accounts,Chief Accounts Officer,Mr. Ramesh Babu,accounts@college.edu,+91-040-12345690,Admin Block Room 201
Placements,Training and Placement Officer,Dr. Srinivas Reddy,placements@college.edu,+91-040-12345700,Placement Block Room 101
Hostel,Boys Hostel Warden,Mr. Vijay Kumar,boyshostel@college.edu,+91-9876543210,Boys Hostel Block
Hostel,Girls Hostel Warden,Ms. Sunitha Reddy,girlshostel@college.edu,+91-9876543211,Girls Hostel Block
Library,Chief Librarian,Dr. Madhavi Latha,library@college.edu,+91-040-12345720,Central Library
CSE,HOD Computer Science,Dr. Ramana Murthy,hod.cse@college.edu,+91-040-12345800,CSE Block Room 301
Principal,Principal,Dr. Bhaskara Rao,principal@college.edu,+91-040-12345950,Admin Block Room 401"""
    
    with open("college_data/contact_info.csv", "w", encoding='utf-8') as f:
        f.write(contact_csv)
    print("✅ Created: contact_info.csv (8 rows)")
    
    print("="*60)
    print("✅ SUCCESS! All 10 CSV files created in 'college_data/' folder")
    print("\n📊 Summary:")
    print("   - fees.csv (31 rows)")
    print("   - courses.csv (7 rows)")
    print("   - eligibility.csv (8 rows)")
    print("   - specializations.csv (15 rows)")
    print("   - admission_process.csv (12 rows)")
    print("   - facilities.csv (10 rows)")
    print("   - placements.csv (6 rows)")
    print("   - scholarships.csv (6 rows)")
    print("   - faq.csv (8 rows)")
    print("   - contact_info.csv (8 rows)")
    print("\n🚀 Next steps:")
    print("   1. Run: python preload_documents.py")
    print("   2. Run: python main.py")
    print("   3. Run: python app_gradio_student.py")
    print("   4. Start asking questions!")

if __name__== "__main__":
    create_college_data_files()