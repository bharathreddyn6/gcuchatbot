"""
College Chatbot - Modular Admin Panel (Gradio)
For administrators to manage chatbot modules:
- Library: Database connection configuration
- Study Plan: Timetable uploads (AI auto-detects branch/semester)
- Academic: Portal credentials
- Complaints: Complaint categories and database
Password protected
"""

import gradio as gr
import requests
import json
import os
from typing import List, Dict, Any
from datetime import datetime
from pathlib import Path

from module_config import config_manager, MODULE_DATA_DIR

# API Configuration
API_BASE = "http://localhost:8000"

# Simple password protection (change this!)
ADMIN_PASSWORD = "admin123"  # CHANGE THIS IN PRODUCTION!


def check_password(password: str) -> bool:
    """Verify admin password"""
    return password == ADMIN_PASSWORD


# ==================== Dashboard Functions ====================

def get_dashboard_overview(password: str) -> str:
    """Get overview of all modules and system status"""
    if not check_password(password):
        return "❌ **Incorrect Password!** Access denied."
    
    modules = config_manager.get_all_modules()
    
    output = """## 📊 System Dashboard

### Module Status
"""
    for module_id, info in modules.items():
        status = "🟢 Enabled" if info['enabled'] else "⚪ Disabled"
        has_config = "✅ Configured" if info['config'] else "⚠️ Not configured"
        last_update = info.get('last_updated', 'Never')[:10] if info.get('last_updated') else 'Never'
        
        output += f"""
**{info['icon']} {info['name']}**
- Status: {status}
- Configuration: {has_config}
- Last Updated: {last_update}
"""
    
    # Study Plan Files
    files = config_manager.get_study_plan_files()
    output += f"""
---
### 📖 Study Plan Files
- **Total Files Uploaded:** {len(files)}
"""
    
    # Try to get backend stats
    try:
        response = requests.get(f"{API_BASE}/stats", timeout=5)
        if response.status_code == 200:
            stats = response.json()
            output += f"""
---
### 💬 Backend Status
- **Vector Store Documents:** {stats.get('total_documents', 'N/A')}
- **Active Sessions:** {stats.get('active_conversations', 0)}
- **Feedback Received:** {stats.get('feedback_received', 0)}
"""
    except:
        output += "\n⚠️ Backend not reachable at " + API_BASE
    
    return output


# ==================== Library Functions ====================

def get_library_config(password: str) -> tuple:
    """Get current library configuration"""
    if not check_password(password):
        return "sqlite", "", "", "", "", "", "❌ Incorrect password"
    
    config = config_manager.get_module_config("library") or {}
    return (
        config.get("driver", "sqlite"),
        config.get("host", ""),
        str(config.get("port", "")),
        config.get("database", ""),
        config.get("username", ""),
        config.get("password", ""),
        "✅ Configuration loaded" if config else "⚠️ No configuration saved"
    )


def save_library_config(password: str, driver: str, host: str, port: str, 
                         database: str, username: str, db_password: str) -> str:
    """Save library database configuration"""
    if not check_password(password):
        return "❌ **Incorrect Password!** Access denied."
    
    config = {
        "driver": driver,
        "host": host,
        "port": int(port) if port else None,
        "database": database,
        "username": username,
        "password": db_password
    }
    
    if driver == "sqlite":
        config["file_path"] = database
    
    config_manager.set_module_config("library", config)
    return f"✅ **Library configuration saved!**\n\nDriver: {driver}\nDatabase: {database}"


def test_library_connection(password: str, driver: str, host: str, port: str,
                             database: str, username: str, db_password: str) -> str:
    """Test library database connection"""
    if not check_password(password):
        return "❌ **Incorrect Password!** Access denied."
    
    config = {
        "driver": driver,
        "host": host,
        "port": int(port) if port else None,
        "database": database,
        "username": username,
        "password": db_password,
        "file_path": database if driver == "sqlite" else None
    }
    
    result = config_manager.test_database_connection(config)
    
    if result["success"]:
        return f"🟢 **Connection Successful!**\n\n{result['message']}"
    else:
        return f"🔴 **Connection Failed!**\n\n{result['message']}"


# ==================== Study Plan Functions ====================

def upload_study_plan_files(password: str, files: List[str]) -> str:
    """Upload timetable/study plan files"""
    if not check_password(password):
        return "❌ **Incorrect Password!** Access denied."
    
    if not files:
        return "⚠️ No files selected"
    
    results = []
    for file_path in files:
        filename = os.path.basename(file_path)
        result = config_manager.save_study_plan_file(file_path, filename)
        
        if result["success"]:
            results.append(f"✅ {filename}")
        else:
            results.append(f"❌ {filename}: {result['message']}")
    
    return f"""📁 **Upload Results**

{chr(10).join(results)}

**Note:** AI will automatically analyze these files to detect branch, semester, and schedule information when students ask questions.
"""


def get_study_plan_files_list(password: str) -> str:
    """Get list of uploaded study plan files"""
    if not check_password(password):
        return "❌ **Incorrect Password!** Access denied."
    
    files = config_manager.get_study_plan_files()
    
    if not files:
        return "📁 No files uploaded yet. Upload timetables (PDF/CSV/images) to enable study plan generation."
    
    output = "## 📁 Uploaded Study Plan Files\n\n"
    
    for idx, file_info in enumerate(files, 1):
        status_emoji = "🔄" if file_info.get("status") == "pending_analysis" else "✅"
        size_kb = file_info.get("file_size", 0) / 1024
        uploaded = file_info.get("uploaded_at", "")[:16].replace("T", " ")
        
        detected = file_info.get("detected_metadata", {})
        branch = detected.get("branch", "Auto-detect pending")
        semester = detected.get("semester", "Auto-detect pending")
        
        output += f"""**{idx}. {file_info.get('original_filename', 'Unknown')}**
   - {status_emoji} Status: {file_info.get('status', 'unknown')}
   - 📅 Uploaded: {uploaded}
   - 📦 Size: {size_kb:.1f} KB
   - 🎓 Branch: {branch}
   - 📚 Semester: {semester}

"""
    
    return output


def delete_study_file(password: str, filename: str) -> str:
    """Delete a study plan file"""
    if not check_password(password):
        return "❌ **Incorrect Password!** Access denied."
    
    if not filename.strip():
        return "⚠️ Please enter a filename to delete"
    
    result = config_manager.delete_study_plan_file(filename.strip())
    
    if result["success"]:
        return f"🗑️ **Deleted:** {filename}"
    else:
        return f"❌ {result['message']}"


# ==================== Academic Functions ====================

def get_academic_config(password: str) -> tuple:
    """Get current academic portal configuration"""
    if not check_password(password):
        return "", "", "", "❌ Incorrect password"
    
    config = config_manager.get_module_config("academic") or {}
    return (
        config.get("portal_url", "https://grms.gcu.edu.in"),
        config.get("username", ""),
        config.get("password", ""),
        "✅ Configuration loaded" if config else "⚠️ Default settings"
    )


def save_academic_config(password: str, portal_url: str, username: str, portal_password: str) -> str:
    """Save academic portal configuration"""
    if not check_password(password):
        return "❌ **Incorrect Password!** Access denied."
    
    config = {
        "portal_url": portal_url,
        "username": username,
        "password": portal_password
    }
    
    config_manager.set_module_config("academic", config)
    return f"""✅ **Academic Portal Configuration Saved!**

Portal URL: {portal_url}
Username: {username}
Password: {'*' * len(portal_password)}

This configuration will be used when students query their attendance, results, etc.
"""


# ==================== Complaints Functions ====================

def get_complaints_config(password: str) -> tuple:
    """Get current complaints configuration"""
    if not check_password(password):
        return "sqlite", "", "", "", "", "", [], "❌ Incorrect password"
    
    config = config_manager.get_module_config("complaints") or {}
    module_def = config_manager.get_module_definition("complaints")
    default_categories = module_def.get("categories", [])
    
    return (
        config.get("driver", "sqlite"),
        config.get("host", ""),
        str(config.get("port", "")),
        config.get("database", "complaints.db"),
        config.get("username", ""),
        config.get("password", ""),
        config.get("categories", default_categories),
        "✅ Configuration loaded" if config.get("database") else "⚠️ Using defaults"
    )


def save_complaints_config(password: str, driver: str, host: str, port: str,
                           database: str, username: str, db_password: str,
                           categories: List[str]) -> str:
    """Save complaints configuration"""
    if not check_password(password):
        return "❌ **Incorrect Password!** Access denied."
    
    config = {
        "driver": driver,
        "host": host,
        "port": int(port) if port else None,
        "database": database,
        "username": username,
        "password": db_password,
        "categories": categories if categories else ["Infrastructure", "Academic", "Hostel", "Transport", "Other"]
    }
    
    config_manager.set_module_config("complaints", config)
    return f"""✅ **Complaints Configuration Saved!**

Driver: {driver}
Database: {database}
Categories: {', '.join(config['categories'])}
"""


# ==================== System Functions ====================

def get_statistics(password: str) -> str:
    """Get system statistics"""
    if not check_password(password):
        return "❌ **Incorrect Password!** Access denied."
    
    try:
        response = requests.get(f"{API_BASE}/stats")
        if response.status_code == 200:
            stats = response.json()
            
            total_feedback = stats.get('positive_feedback', 0) + stats.get('negative_feedback', 0)
            satisfaction = (stats.get('positive_feedback', 0) / total_feedback * 100) if total_feedback > 0 else 0
            
            return f"""## 📊 System Statistics

### 📚 Knowledge Base
- **Total Documents:** {stats.get('total_documents', 0)}
- **Vector Store:** {stats.get('vector_store_type', 'N/A')}
- **Embedding Model:** {stats.get('embedding_model', 'N/A')}

### 💬 Usage Metrics
- **Active Conversations:** {stats.get('active_conversations', 0)}
- **Total Feedback:** {total_feedback}

### 😊 Satisfaction
- **👍 Positive:** {stats.get('positive_feedback', 0)}
- **👎 Negative:** {stats.get('negative_feedback', 0)}
- **Rate:** {satisfaction:.1f}%

### ⚙️ System Health
- **Status:** 🟢 Healthy
- **Last Updated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
"""
        else:
            return f"❌ Failed to fetch statistics: {response.status_code}"
    except Exception as e:
        return f"❌ Error: {str(e)}"


def upload_general_files(password: str, files: List[str]) -> str:
    """Upload general knowledge base files (PDF/CSV)"""
    if not check_password(password):
        return "❌ **Incorrect Password!** Access denied."
    
    if not files:
        return "⚠️ No files selected"
    
    try:
        file_objects = []
        for file_path in files:
            file_objects.append(('files', open(file_path, 'rb')))
        
        response = requests.post(f"{API_BASE}/upload", files=file_objects)
        
        for _, f in file_objects:
            f.close()
        
        if response.status_code == 200:
            result = response.json()
            return f"""✅ **Upload Successful!**

📁 **Files:** {', '.join(result['uploaded_files'])}
📊 **Chunks Created:** {result['chunks_created']}
💬 **Message:** {result['message']}
"""
        else:
            return f"❌ **Upload Failed:** {response.status_code}\n\n{response.text}"
    
    except Exception as e:
        return f"❌ **Error:** {str(e)}\n\nMake sure backend is running at {API_BASE}"


# ==================== Build Admin Panel ====================

with gr.Blocks(
    theme=gr.themes.Soft(primary_hue="blue", secondary_hue="purple"),
    title="College Chatbot - Admin Panel",
    css="""
    .gradio-container {max-width: 1200px !important;}
    footer {display: none !important;}
    .module-card {padding: 15px; border-radius: 10px; margin: 10px 0;}
    """
) as admin:
    
    # Header
    gr.Markdown("""
    # 🔐 College Chatbot - Modular Admin Panel
    ### Manage modules, configure data sources, and monitor system health
    
    ⚠️ **Authorized Personnel Only**
    """)
    
    # Password input (shared across tabs)
    password_input = gr.Textbox(
        label="Admin Password",
        type="password",
        placeholder="Enter admin password",
        value=""
    )
    
    gr.Markdown("---")
    
    with gr.Tabs():
        
        # Tab 1: Dashboard
        with gr.Tab("🏠 Dashboard"):
            gr.Markdown("### System Overview")
            dashboard_output = gr.Markdown()
            refresh_dashboard_btn = gr.Button("🔄 Refresh Dashboard", variant="primary")
            
            refresh_dashboard_btn.click(
                get_dashboard_overview,
                inputs=[password_input],
                outputs=[dashboard_output]
            )
        
        # Tab 2: Library Configuration
        with gr.Tab("📚 Library"):
            gr.Markdown("""
            ### Library Database Configuration
            Configure the database connection for library queries (book search, availability, etc.)
            """)
            
            with gr.Row():
                lib_driver = gr.Dropdown(
                    choices=["sqlite", "mysql", "postgresql"],
                    value="sqlite",
                    label="Database Driver"
                )
                lib_host = gr.Textbox(label="Host", placeholder="localhost (not needed for SQLite)")
                lib_port = gr.Textbox(label="Port", placeholder="3306 / 5432")
            
            with gr.Row():
                lib_database = gr.Textbox(label="Database Name / File Path", placeholder="library.db or library_database")
                lib_username = gr.Textbox(label="Username", placeholder="(not needed for SQLite)")
                lib_password = gr.Textbox(label="Password", type="password")
            
            lib_status = gr.Markdown()
            
            with gr.Row():
                lib_load_btn = gr.Button("📥 Load Current Config")
                lib_test_btn = gr.Button("🔌 Test Connection", variant="secondary")
                lib_save_btn = gr.Button("💾 Save Configuration", variant="primary")
            
            lib_result = gr.Markdown()
            
            lib_load_btn.click(
                get_library_config,
                inputs=[password_input],
                outputs=[lib_driver, lib_host, lib_port, lib_database, lib_username, lib_password, lib_status]
            )
            
            lib_test_btn.click(
                test_library_connection,
                inputs=[password_input, lib_driver, lib_host, lib_port, lib_database, lib_username, lib_password],
                outputs=[lib_result]
            )
            
            lib_save_btn.click(
                save_library_config,
                inputs=[password_input, lib_driver, lib_host, lib_port, lib_database, lib_username, lib_password],
                outputs=[lib_result]
            )
        
        # Tab 3: Study Plan
        with gr.Tab("📖 Study Plan"):
            gr.Markdown("""
            ### Study Plan / Timetable Management
            Upload timetables (PDF, CSV, Images). AI will automatically detect:
            - Branch (CSE, ECE, ME, etc.)
            - Semester
            - Schedule information
            
            When students ask for study plans, AI will analyze these files to generate personalized recommendations.
            """)
            
            with gr.Row():
                with gr.Column(scale=2):
                    study_file_upload = gr.File(
                        label="Upload Timetable Files",
                        file_count="multiple",
                        file_types=[".pdf", ".csv", ".png", ".jpg", ".jpeg"],
                        type="filepath"
                    )
                    study_upload_btn = gr.Button("🚀 Upload Files", variant="primary", size="lg")
                    study_upload_result = gr.Markdown()
                
                with gr.Column(scale=3):
                    study_files_list = gr.Markdown()
                    study_refresh_btn = gr.Button("🔄 Refresh File List")
            
            gr.Markdown("---")
            gr.Markdown("### Delete File")
            with gr.Row():
                delete_filename = gr.Textbox(label="Filename to Delete", placeholder="Enter exact saved filename")
                delete_btn = gr.Button("🗑️ Delete", variant="stop")
            delete_result = gr.Markdown()
            
            study_upload_btn.click(
                upload_study_plan_files,
                inputs=[password_input, study_file_upload],
                outputs=[study_upload_result]
            )
            
            study_refresh_btn.click(
                get_study_plan_files_list,
                inputs=[password_input],
                outputs=[study_files_list]
            )
            
            delete_btn.click(
                delete_study_file,
                inputs=[password_input, delete_filename],
                outputs=[delete_result]
            )
        
        # Tab 4: Academic
        with gr.Tab("🎓 Academic"):
            gr.Markdown("""
            ### Academic Portal Configuration
            Configure student portal credentials for attendance, results, and other academic data.
            """)
            
            acad_portal_url = gr.Textbox(label="Portal URL", placeholder="https://grms.gcu.edu.in")
            acad_username = gr.Textbox(label="Default Username", placeholder="student@gcu.edu.in")
            acad_password = gr.Textbox(label="Default Password", type="password")
            
            acad_status = gr.Markdown()
            
            with gr.Row():
                acad_load_btn = gr.Button("📥 Load Current Config")
                acad_save_btn = gr.Button("💾 Save Configuration", variant="primary")
            
            acad_result = gr.Markdown()
            
            acad_load_btn.click(
                get_academic_config,
                inputs=[password_input],
                outputs=[acad_portal_url, acad_username, acad_password, acad_status]
            )
            
            acad_save_btn.click(
                save_academic_config,
                inputs=[password_input, acad_portal_url, acad_username, acad_password],
                outputs=[acad_result]
            )
        
        # Tab 5: Complaints
        with gr.Tab("📝 Complaints"):
            gr.Markdown("""
            ### Complaints System Configuration
            Configure database and categories for student complaints.
            """)
            
            with gr.Row():
                comp_driver = gr.Dropdown(
                    choices=["sqlite", "mysql", "postgresql"],
                    value="sqlite",
                    label="Database Driver"
                )
                comp_host = gr.Textbox(label="Host")
                comp_port = gr.Textbox(label="Port")
            
            with gr.Row():
                comp_database = gr.Textbox(label="Database Name / File Path", value="complaints.db")
                comp_username = gr.Textbox(label="Username")
                comp_password = gr.Textbox(label="Password", type="password")
            
            comp_categories = gr.Dropdown(
                choices=["Infrastructure", "Academic", "Hostel", "Transport", "Canteen", "Other"],
                multiselect=True,
                label="Complaint Categories",
                value=["Infrastructure", "Academic", "Hostel", "Transport", "Other"]
            )
            
            comp_status = gr.Markdown()
            
            with gr.Row():
                comp_load_btn = gr.Button("📥 Load Current Config")
                comp_save_btn = gr.Button("💾 Save Configuration", variant="primary")
            
            comp_result = gr.Markdown()
            
            comp_load_btn.click(
                get_complaints_config,
                inputs=[password_input],
                outputs=[comp_driver, comp_host, comp_port, comp_database, comp_username, comp_password, comp_categories, comp_status]
            )
            
            comp_save_btn.click(
                save_complaints_config,
                inputs=[password_input, comp_driver, comp_host, comp_port, comp_database, comp_username, comp_password, comp_categories],
                outputs=[comp_result]
            )
        
        # Tab 6: Knowledge Base Upload
        with gr.Tab("📤 Knowledge Base"):
            gr.Markdown("""
            ### Upload College Documents
            Upload general knowledge base documents (prospectus, syllabus, fee structure, etc.)
            These will be used for general college queries.
            
            **Supported Formats:** PDF, CSV
            """)
            
            kb_file_upload = gr.File(
                label="Select Files to Upload",
                file_count="multiple",
                file_types=[".pdf", ".csv"],
                type="filepath"
            )
            
            kb_upload_btn = gr.Button("🚀 Upload to Knowledge Base", variant="primary", size="lg")
            kb_upload_output = gr.Markdown()
            
            kb_upload_btn.click(
                upload_general_files,
                inputs=[password_input, kb_file_upload],
                outputs=[kb_upload_output]
            )
        
        # Tab 7: Analytics
        with gr.Tab("📊 Analytics"):
            gr.Markdown("### System Statistics and Analytics")
            
            stats_output = gr.Markdown()
            refresh_stats_btn = gr.Button("🔄 Refresh Statistics", variant="primary")
            
            refresh_stats_btn.click(
                get_statistics,
                inputs=[password_input],
                outputs=[stats_output]
            )
        
        # Tab 8: Help
        with gr.Tab("❓ Help"):
            gr.Markdown("""
## 📖 Admin Panel Guide

### 🔐 Security
- **Default Password:** `admin123`
- **⚠️ IMPORTANT:** Change the password before deployment!

---

### 📚 Library Module
Configure database connection for library queries:
1. Select driver (SQLite for simplicity, MySQL/PostgreSQL for production)
2. Enter connection details
3. Click "Test Connection" to verify
4. Save configuration

---

### 📖 Study Plan Module
Upload timetables for AI-powered study plan generation:
1. Upload PDF/CSV/Image files containing timetables
2. AI automatically detects branch, semester, and schedule
3. Students can ask "Create a study plan" and AI will analyze these files

---

### 🎓 Academic Module
Configure student portal for live data:
1. Enter portal URL (e.g., https://grms.gcu.edu.in)
2. Enter default credentials
3. Students can query attendance, results, etc.

---

### 📝 Complaints Module
Set up complaint system:
1. Configure database for storing complaints
2. Define complaint categories
3. Students can file and track complaints

---

### 🔧 Troubleshooting

**Problem:** Database connection fails  
**Solution:** Check credentials, ensure database server is running

**Problem:** File upload fails  
**Solution:** Check file format (PDF/CSV only), ensure backend is running

**Problem:** Backend not reachable  
**Solution:** Start backend with `python ragmain.py`
            """)
    
    # Footer
    gr.Markdown("""
    ---
    🔒 **Security Notice:** Change the default password before deploying to production.
    
    🔗 **Related Services:**
    - Student Chatbot: http://localhost:8501 (Streamlit)
    - Backend API: http://localhost:8000
    - API Docs: http://localhost:8000/docs
    """)


if __name__ == "__main__":
    print("🔐 Starting Modular Admin Panel...")
    print("🌐 Opening at: http://localhost:7861")
    print("🔑 Default Password: admin123 (CHANGE THIS!)")
    print("\n⚠️  Make sure backend is running: python ragmain.py\n")
    
    admin.launch(
        server_name="0.0.0.0",
        server_port=7861,
        share=True,
        show_error=True
    )