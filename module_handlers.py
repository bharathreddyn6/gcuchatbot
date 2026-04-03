"""
Module Handlers for College Chatbot
Handles queries routed to specific modules: Library, Study Plan, Academic, Complaints
"""

import os
import json
from typing import Dict, Any, List, Optional
from datetime import datetime
from pathlib import Path

# AI/LLM imports
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain.schema import Document

from module_config import config_manager, MODULE_DATA_DIR

# Initialize LLM for AI analysis
try:
    llm = ChatGoogleGenerativeAI(
        model="gemini-2.0-flash",
        temperature=0.3,
        convert_system_message_to_human=True
    )
except Exception as e:
    print(f"[WARN] Could not initialize Gemini LLM: {e}")
    llm = None


# ==================== Library Handler ====================

class LibraryHandler:
    """Handles library-related queries using configured database"""
    
    def __init__(self):
        self.config = config_manager.get_module_config("library")
    
    def reload_config(self):
        """Reload configuration from file"""
        self.config = config_manager.get_module_config("library")
    
    def _get_connection(self):
        """Get database connection based on config"""
        if not self.config:
            return None
        
        driver = self.config.get("driver", "sqlite")
        
        try:
            if driver == "sqlite":
                import sqlite3
                db_path = self.config.get("file_path") or self.config.get("database", "library.db")
                return sqlite3.connect(db_path)
            
            elif driver == "mysql":
                import mysql.connector
                return mysql.connector.connect(
                    host=self.config.get("host", "localhost"),
                    port=int(self.config.get("port", 3306)),
                    database=self.config.get("database"),
                    user=self.config.get("username"),
                    password=self.config.get("password")
                )
            
            elif driver == "postgresql":
                import psycopg2
                return psycopg2.connect(
                    host=self.config.get("host", "localhost"),
                    port=int(self.config.get("port", 5432)),
                    database=self.config.get("database"),
                    user=self.config.get("username"),
                    password=self.config.get("password")
                )
        except Exception as e:
            print(f"[ERROR] Library DB connection failed: {e}")
            return None
    
    async def handle_query(self, query: str) -> Dict[str, Any]:
        """Handle library-related query"""
        self.reload_config()
        
        if not self.config:
            return {
                "answer": "📚 Library module is not configured. Please ask an administrator to set up the library database connection.",
                "sources": ["Library Module"]
            }
        
        conn = self._get_connection()
        if not conn:
            return {
                "answer": "📚 Could not connect to the library database. Please contact an administrator.",
                "sources": ["Library Module"]
            }
        
        try:
            # Get table info for context
            cursor = conn.cursor()
            
            # Check if it's SQLite or MySQL/PostgreSQL
            driver = self.config.get("driver", "sqlite")
            
            if driver == "sqlite":
                cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
            else:
                cursor.execute("SHOW TABLES;")
            
            tables = [row[0] for row in cursor.fetchall()]
            
            if not tables:
                conn.close()
                return {
                    "answer": "📚 The library database appears to be empty. No tables found.",
                    "sources": ["Library Database"]
                }
            
            # Use AI to generate SQL query if LLM is available
            if llm:
                # Get schema for first few tables
                schema_info = []
                for table in tables[:5]:
                    if driver == "sqlite":
                        cursor.execute(f"PRAGMA table_info({table});")
                        columns = [row[1] for row in cursor.fetchall()]
                    else:
                        cursor.execute(f"DESCRIBE {table};")
                        columns = [row[0] for row in cursor.fetchall()]
                    schema_info.append(f"{table}: {', '.join(columns)}")
                
                schema_str = "\n".join(schema_info)
                
                # Generate SQL with AI
                sql_prompt = PromptTemplate(
                    input_variables=["query", "schema"],
                    template="""You are a library database assistant. Given the user's question and database schema, generate a safe SELECT SQL query.

Database Schema:
{schema}

User Question: {query}

Common abbreviations to expand:
- CSE = Computer Science
- ECE = Electronics
- IT = Information Technology  
- AI/ML = AI/ML or Artificial Intelligence
- DSA = Data Structures

Rules:
- Only generate SELECT queries (no INSERT, UPDATE, DELETE)
- Use LIKE with % wildcards for flexible matching (e.g., category LIKE '%Computer%')
- Return ONLY the raw SQL query - NO markdown, NO code blocks, NO backticks
- Do NOT wrap the query in ```sql or any formatting
- For "show all" type queries, use: SELECT * FROM books
- If you can't generate a valid query, return: SELECT * FROM books

SQL Query:"""
                )
                
                chain = LLMChain(llm=llm, prompt=sql_prompt)
                sql_query = chain.run(query=query, schema=schema_str).strip()
                
                # Clean up any markdown formatting the AI might have added
                sql_query = sql_query.replace("```sql", "").replace("```", "").strip()
                
                # Execute query
                try:
                    cursor.execute(sql_query)
                    results = cursor.fetchall()
                    
                    if results:
                        # Format results
                        columns = [desc[0] for desc in cursor.description]
                        formatted = []
                        for row in results[:10]:  # Limit to 10 results
                            row_dict = dict(zip(columns, row))
                            formatted.append(str(row_dict))
                        
                        answer = f"📚 **Library Search Results:**\n\n" + "\n".join(formatted)
                        if len(results) > 10:
                            answer += f"\n\n(Showing 10 of {len(results)} results)"
                    else:
                        answer = "📚 No results found for your query."
                    
                    conn.close()
                    return {
                        "answer": answer,
                        "sources": ["Library Database"],
                        "sql_query": sql_query
                    }
                
                except Exception as e:
                    conn.close()
                    return {
                        "answer": f"📚 I couldn't find that information in the library database. Error: {str(e)}",
                        "sources": ["Library Database"]
                    }
            
            else:
                # No LLM, just return basic info
                conn.close()
                return {
                    "answer": f"📚 Library database connected! Available tables: {', '.join(tables)}. AI query processing is not available.",
                    "sources": ["Library Database"]
                }
        
        except Exception as e:
            return {
                "answer": f"📚 Error querying library database: {str(e)}",
                "sources": ["Library Module"]
            }


# ==================== Study Plan Handler ====================

class StudyPlanHandler:
    """Handles study plan queries by analyzing uploaded timetables"""
    
    def __init__(self):
        self.upload_dir = MODULE_DATA_DIR / "study_plan" / "uploads"
    
    def _get_uploaded_files(self) -> List[Dict[str, Any]]:
        """Get list of uploaded timetable files with metadata"""
        return config_manager.get_study_plan_files()
    
    def _extract_text_from_file(self, file_path: str) -> str:
        """Extract text content from PDF, CSV, or image file"""
        file_path = Path(file_path)
        
        if not file_path.exists():
            return ""
        
        suffix = file_path.suffix.lower()
        
        try:
            if suffix == ".csv":
                import pandas as pd
                df = pd.read_csv(file_path)
                return df.to_string()
            
            elif suffix == ".pdf":
                from langchain_community.document_loaders import PyPDFLoader
                loader = PyPDFLoader(str(file_path))
                pages = loader.load()
                return "\n".join([p.page_content for p in pages])
            
            elif suffix in [".png", ".jpg", ".jpeg"]:
                # For images, we'll use AI vision if available
                # For now, return placeholder
                return f"[Image file: {file_path.name} - AI vision analysis required]"
            
            else:
                return ""
        
        except Exception as e:
            print(f"[WARN] Could not extract text from {file_path}: {e}")
            return ""
    
    async def handle_query(self, query: str, branch: str = None, semester: str = None) -> Dict[str, Any]:
        """
        Handle study plan query.
        If branch/semester not provided, ask user first before generating plan.
        """
        files = self._get_uploaded_files()
        
        if not files:
            return {
                "answer": """📖 **Study Plan Module**

No timetables have been uploaded yet. Please ask an administrator to upload:
- Class timetables (PDF/CSV/Image)
- Exam schedules
- Academic calendar

Once uploaded, I can:
- Create personalized study plans
- Suggest optimal study times
- Help with exam preparation schedules
""",
                "sources": ["Study Plan Module"]
            }
        
        # Try to detect branch and semester from the query
        query_lower = query.lower()
        
        # Common branch names
        branches = ["cse", "ece", "eee", "me", "ce", "civil", "it", "mba", "mca", 
                   "computer science", "electronics", "mechanical", "electrical"]
        detected_branch = None
        for b in branches:
            if b in query_lower:
                detected_branch = b.upper()
                break
        
        # Detect semester
        detected_semester = None
        import re
        sem_match = re.search(r'(\d+)\s*(?:st|nd|rd|th)?\s*sem(?:ester)?', query_lower)
        if sem_match:
            detected_semester = sem_match.group(1)
        else:
            # Check for words like "first", "second", etc.
            sem_words = {"first": "1", "second": "2", "third": "3", "fourth": "4", 
                        "fifth": "5", "sixth": "6", "seventh": "7", "eighth": "8"}
            for word, num in sem_words.items():
                if word in query_lower and "sem" in query_lower:
                    detected_semester = num
                    break
        
        # Use provided values or detected values
        branch = branch or detected_branch
        semester = semester or detected_semester
        
        # Detect section from query (A, B, C)
        detected_section = None
        section_match = re.search(r'\b([abc])\s*section\b|\bsection\s*([abc])\b|\b([abc])\b\s*$', query_lower)
        if section_match:
            detected_section = (section_match.group(1) or section_match.group(2) or section_match.group(3)).upper()
        
        # If branch or semester not available, ASK the user first
        if not branch or not semester:
            # Get available branches from uploaded files
            available_info = []
            for file_info in files[:5]:
                available_info.append(f"- {file_info.get('original_filename', 'Unknown')}")
            
            missing = []
            if not branch:
                missing.append("**Branch** (e.g., CSE, ECE, ME, IT, MBA, MCA)")
            if not semester:
                missing.append("**Semester** (e.g., 1st, 2nd, 3rd, 4th, 5th, 6th, 7th, 8th)")
            
            return {
                "answer": f"""📖 **Study Plan Assistant**

To create a personalized study plan, I need to know:

{chr(10).join(missing)}

**Example:** "Create a study plan for CSE 5th semester section A"

📁 **Available Timetables:**
{chr(10).join(available_info)}

Please specify your branch and semester so I can analyze the correct timetable for you!
""",
                "sources": ["Study Plan Module"]
            }
        
        # Now we have both branch and semester, proceed with study plan generation
        # FILTER files by semester and section to get the most relevant ones
        relevant_files = []
        other_files = []
        
        for file_info in files:
            filename_lower = file_info.get("original_filename", "").lower()
            
            # Check if file matches semester
            semester_keywords = [f"{semester}th", f"{semester}sem", f"sem{semester}", f"semester{semester}", 
                                f"{semester} sem", f"sem {semester}", f"{semester}nd", f"{semester}rd", f"{semester}st"]
            matches_semester = any(kw in filename_lower.replace(" ", "") for kw in semester_keywords)
            
            # Check if file matches section
            matches_section = detected_section and detected_section.lower() in filename_lower
            
            # Prioritize files
            if matches_semester:
                relevant_files.append((file_info, 2 if matches_section else 1))
            else:
                other_files.append(file_info)
        
        # Sort relevant files by score (higher score = more relevant)
        relevant_files.sort(key=lambda x: x[1], reverse=True)
        
        # Use relevant files first, then others if needed
        files_to_process = [f[0] for f in relevant_files]
        if not files_to_process:
            # No matching files, use all files
            files_to_process = files
        
        # Extract content from selected files
        all_content = []
        processed_files = []
        
        for file_info in files_to_process[:3]:  # Limit to top 3 relevant files
            file_path = self.upload_dir / file_info.get("saved_filename", "")
            if file_path.exists():
                content = self._extract_text_from_file(file_path)
                if content:
                    all_content.append(f"=== File: {file_info.get('original_filename')} ===\n{content}")
                    processed_files.append(file_info.get('original_filename', 'Unknown'))
        
        if not all_content and not llm:
            return {
                "answer": "📖 Files are uploaded but I couldn't extract their content. Please ensure PDF/CSV files are properly formatted.",
                "sources": ["Study Plan Module"]
            }
        
        combined_content = "\n\n".join(all_content)
        
        # Add section info to the query context
        section_info = f" Section {detected_section}" if detected_section else ""
        
        # Use AI to analyze and generate study plan
        if llm:
            study_plan_prompt = PromptTemplate(
                input_variables=["query", "timetable_content", "branch", "semester", "section"],
                template="""You are an expert study plan creator. Create a SMART, CONCISE, and PRACTICAL weekly study plan.

STUDENT INFO:
- Branch: {branch}
- Semester: {semester}  
- Section: {section}

TIMETABLE DATA:
{timetable_content}

STRICT RULES:
1. ONLY use data for Section {section} - completely IGNORE all other sections
2. READ the timetable carefully - find the ACTUAL last class time for EACH day (it varies - could be 12 PM, 2 PM, 4 PM, etc.)
3. Add 2-3 hours REST after the last class ends, THEN start study time
4. Study sessions should be 1-2 hours per subject MAX with 15-min breaks

WEEKLY PATTERN (MANDATORY):
- Monday-Friday: Study only AFTER classes + rest time
- Saturday: LIGHT study (2-3 hours max in morning only)  
- Sunday: FULL DAY OFF, only evening revision (7-9 PM) for Monday's subjects

OUTPUT FORMAT (BE CONCISE - use this EXACT format):

📅 **WEEKLY STUDY SCHEDULE**

**Your Subjects:** [List only subjects from Section {section}]

**Monday:**
Classes end: [time]
Free time: [end time] - [study start]
📚 [Study start] - [end]: [Subject] - [specific topic/activity]
📚 [time] - [time]: [Subject] - [specific topic/activity]

**Tuesday:**
[Same format]

[Continue for each day...]

**Saturday:**
🌅 Morning only (10 AM - 1 PM): Light revision of weak topics
🎮 Rest of day: FREE

**Sunday:**
☀️ Full day: REST & RECHARGE
🌙 7:00 PM - 9:00 PM: Quick revision for Monday classes only

---
💡 **Quick Tips:**
- [2-3 brief practical tips only]

DO NOT include:
- Long paragraphs of explanation
- Generic study advice
- Information about other sections
- Assumptions list
- Detailed subject techniques

BE SHORT AND PRACTICAL. Students want a USABLE schedule, not an essay.

Your Response:"""
            )
            
            try:
                chain = LLMChain(llm=llm, prompt=study_plan_prompt)
                response = chain.run(
                    query=query,
                    timetable_content=combined_content[:12000],
                    branch=branch,
                    semester=semester,
                    section=detected_section or "All"
                )
                
                return {
                    "answer": f"📖 **Study Plan for {branch} - Semester {semester}{section_info}**\n\n{response}",
                    "sources": processed_files
                }
            
            except Exception as e:
                return {
                    "answer": f"📖 Error generating study plan: {str(e)}",
                    "sources": ["Study Plan Module"]
                }
        
        else:
            # No LLM, return basic info
            return {
                "answer": f"📖 I found {len(files_to_process)} relevant timetable files for {branch} Semester {semester}{section_info}, but AI analysis is not available. Files: " + 
                         ", ".join(processed_files),
                "sources": ["Study Plan Module"]
            }
    
    async def analyze_file_metadata(self, file_path: str) -> Dict[str, Any]:
        """Use AI to detect branch, semester, and other metadata from file content"""
        content = self._extract_text_from_file(file_path)
        
        if not content or not llm:
            return {}
        
        metadata_prompt = PromptTemplate(
            input_variables=["content"],
            template="""Analyze this timetable/schedule content and extract metadata.

Content:
{content}

Extract the following in JSON format:
{{
    "branch": "detected branch name (e.g., CSE, ECE, ME) or null",
    "semester": "detected semester number or null",
    "academic_year": "detected year or null",
    "type": "type of schedule (timetable, exam_schedule, academic_calendar, other)",
    "subjects": ["list of subjects mentioned"]
}}

Return ONLY the JSON, nothing else:"""
        )
        
        try:
            chain = LLMChain(llm=llm, prompt=metadata_prompt)
            result = chain.run(content=content[:4000])
            
            # Parse JSON
            import re
            json_match = re.search(r'\{.*\}', result, re.DOTALL)
            if json_match:
                return json.loads(json_match.group())
        
        except Exception as e:
            print(f"[WARN] Could not analyze file metadata: {e}")
        
        return {}


# ==================== Complaints Handler ====================

class ComplaintsHandler:
    """Handles student complaints with DB integration and portal redirect."""

    PORTAL_URL = "http://localhost:8000/complaints"

    def __init__(self):
        self.config = config_manager.get_module_config("complaints")

    def reload_config(self):
        self.config = config_manager.get_module_config("complaints")

    def _extract_complaint_id(self, text: str) -> str | None:
        """Extract a CMP-XXXXXXXX style complaint ID from text."""
        import re
        match = re.search(r'\bCMP-[A-F0-9]{8}\b', text.upper())
        return match.group(0) if match else None

    async def handle_query(self, query: str, session_id: str = None) -> Dict[str, Any]:
        """Handle complaint-related query with smart intent routing."""
        self.reload_config()
        module_def = config_manager.get_module_definition("complaints")
        categories = self.config.get("categories", module_def.get("categories", []))
        query_lower = query.lower()

        # ── SUBMIT INTENT ──────────────────────────────────────────────────────
        submit_words = ["file", "submit", "register", "raise", "lodge",
                        "report", "complain", "new complaint"]
        if any(w in query_lower for w in submit_words):
            return {
                "answer": (
                    "📝 **File a Complaint**\n\n"
                    "To submit your complaint, please visit our dedicated complaint portal:\n\n"
                    f"🔗 **[Open Complaint Portal]({self.PORTAL_URL})**\n\n"
                    "On the portal you can:\n"
                    "- Describe your issue with photos\n"
                    "- Share your location automatically\n"
                    "- Get an instant complaint ID for tracking\n\n"
                    "Your complaint will be automatically routed to the right department! 🚀"
                ),
                "sources": ["Complaints Portal"]
            }

        # ── TRACK INTENT ───────────────────────────────────────────────────────
        track_words = ["status", "track", "where is my", "check complaint", "update"]
        if any(w in query_lower for w in track_words):
            cid = self._extract_complaint_id(query)
            if cid:
                try:
                    from complaint_db import get_complaint
                    record = get_complaint(cid)
                    if record:
                        status = record["status"]
                        status_emoji = {
                            "Pending": "🟡", "In Progress": "🔵",
                            "Resolved": "🟢", "Rejected": "🔴"
                        }.get(status, "⚪")
                        comments_txt = ""
                        if record.get("comments"):
                            last = record["comments"][-1]
                            comments_txt = f"\n\n💬 **Latest Update:** {last['message']} — _{last['author']}_"
                        return {
                            "answer": (
                                f"📝 **Complaint Status: {cid}**\n\n"
                                f"{status_emoji} **Status:** {status}\n"
                                f"🏷️ **Category:** {record.get('category', 'N/A')}\n"
                                f"🏢 **Department:** {record.get('department', 'N/A')}\n"
                                f"📅 **Submitted:** {record['created_at'][:10]}\n"
                                f"📍 **Location:** {record.get('location') or 'Not specified'}"
                                + comments_txt
                            ),
                            "sources": ["Complaints DB"]
                        }
                    else:
                        return {
                            "answer": f"❌ No complaint found with ID **{cid}**. Please check the ID and try again.",
                            "sources": ["Complaints DB"]
                        }
                except Exception as e:
                    print(f"[WARN] Complaint DB lookup failed: {e}")

            return {
                "answer": (
                    "📝 **Track Your Complaint**\n\n"
                    f"Please visit the portal to track your complaint:\n"
                    f"🔗 **[Open Complaint Portal]({self.PORTAL_URL})**\n\n"
                    "Or share your **Complaint ID** (e.g. CMP-AB12CD34) and I'll look it up for you!"
                ),
                "sources": ["Complaints Portal"]
            }

        # ── GENERAL INFO ───────────────────────────────────────────────────────
        return {
            "answer": (
                "📝 **Complaints System**\n\n"
                f"I can help you file or track complaints.\n\n"
                f"🔗 **[Open Complaint Portal]({self.PORTAL_URL})**\n\n"
                f"**Categories:** {', '.join(categories)}\n\n"
                "What would you like to do?\n"
                "- **File a new complaint**\n"
                "- **Track an existing complaint** (share your ID)"
            ),
            "sources": ["Complaints Module"]
        }


# ==================== Module Router ====================

# Initialize handlers
library_handler = LibraryHandler()
study_plan_handler = StudyPlanHandler()
complaints_handler = ComplaintsHandler()


async def route_module_query(context_mode: str, query: str, session_id: str = None, **kwargs) -> Dict[str, Any]:
    """
    Route query to appropriate module handler based on context mode.
    Returns dict with 'answer' and 'sources' keys.
    """
    
    if context_mode == "Library":
        return await library_handler.handle_query(query)
    
    elif context_mode == "Study Plan":
        return await study_plan_handler.handle_query(
            query,
            branch=kwargs.get("branch"),
            semester=kwargs.get("semester")
        )
    
    elif context_mode == "Complaints":
        return await complaints_handler.handle_query(query, session_id)
    
    else:
        # Unknown module
        return {
            "answer": f"Module '{context_mode}' is not configured or recognized.",
            "sources": ["System"]
        }


if __name__ == "__main__":
    import asyncio
    
    async def test():
        print("Testing Module Handlers...")
        
        # Test Library Handler
        print("\n📚 Testing Library Handler...")
        result = await library_handler.handle_query("Is 'Python Programming' book available?")
        print(f"Result: {result['answer'][:200]}...")
        
        # Test Study Plan Handler
        print("\n📖 Testing Study Plan Handler...")
        result = await study_plan_handler.handle_query("Create a study plan for CSE 5th semester")
        print(f"Result: {result['answer'][:200]}...")
        
        # Test Complaints Handler
        print("\n📝 Testing Complaints Handler...")
        result = await complaints_handler.handle_query("I want to file a complaint")
        print(f"Result: {result['answer'][:200]}...")
        
        print("\n✅ All handlers tested!")
    
    asyncio.run(test())
