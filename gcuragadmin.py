"""
College Chatbot - Admin Panel (Gradio)
For college administrators to manage documents and view analytics
Password protected
"""

import gradio as gr
import requests
import json
from typing import List
from datetime import datetime

# API Configuration
API_BASE = "http://localhost:8000"

# Simple password protection (change this!)
ADMIN_PASSWORD = "admin123"  # CHANGE THIS IN PRODUCTION!

def check_password(password: str) -> bool:
    """Verify admin password"""
    return password == ADMIN_PASSWORD

def upload_files(password: str, files: List[str]) -> str:
    """Upload files to the backend"""
    if not check_password(password):
        return "❌ **Incorrect Password!** Access denied."
    
    if not files:
        return "⚠️ No files selected"
    
    try:
        file_objects = []
        for file_path in files:
            file_objects.append(('files', open(file_path, 'rb')))
        
        response = requests.post(f"{API_BASE}/upload", files=file_objects)
        
        # Close files
        for _, f in file_objects:
            f.close()
        
        if response.status_code == 200:
            result = response.json()
            return f"""✅ **Upload Successful!**

📁 **Files Uploaded:** {', '.join(result['uploaded_files'])}
📊 **Chunks Created:** {result['chunks_created']}
💬 **Message:** {result['message']}

The knowledge base has been updated. Students can now ask questions about this content.
"""
        else:
            return f"❌ **Upload Failed:** {response.status_code}\n\n{response.text}"
    
    except Exception as e:
        return f"❌ **Error:** {str(e)}\n\nMake sure the backend is running on {API_BASE}"

def get_statistics(password: str) -> str:
    """Get system statistics"""
    if not check_password(password):
        return "❌ **Incorrect Password!** Access denied."
    
    try:
        response = requests.get(f"{API_BASE}/stats")
        if response.status_code == 200:
            stats = response.json()
            
            # Calculate satisfaction rate
            total_feedback = stats.get('positive_feedback', 0) + stats.get('negative_feedback', 0)
            if total_feedback > 0:
                satisfaction = (stats.get('positive_feedback', 0) / total_feedback) * 100
            else:
                satisfaction = 0
            
            return f"""📊 **System Statistics**

### 📚 Knowledge Base
- **Total Documents:** {stats.get('total_documents', 0)}
- **Vector Store:** {stats.get('vector_store_type', 'N/A')}
- **Embedding Model:** {stats.get('embedding_model', 'N/A')}

### 💬 Usage Metrics
- **Active Conversations:** {stats.get('active_conversations', 0)}
- **Total Feedback Received:** {total_feedback}

### 😊 Satisfaction
- **👍 Positive Feedback:** {stats.get('positive_feedback', 0)}
- **👎 Negative Feedback:** {stats.get('negative_feedback', 0)}
- **Satisfaction Rate:** {satisfaction:.1f}%

### ⚙️ System Health
- **Status:** {"🟢 Healthy" if total_feedback >= 0 else "🔴 Issues"}
- **Last Updated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
"""
        else:
            return f"❌ Failed to fetch statistics: {response.status_code}"
    except Exception as e:
        return f"❌ Error: {str(e)}"

def view_feedback(password: str) -> str:
    """View recent feedback"""
    if not check_password(password):
        return "❌ **Incorrect Password!** Access denied."
    
    try:
        # Read feedback file
        import os
        if not os.path.exists("feedback/feedback.jsonl"):
            return "📝 No feedback received yet."
        
        with open("feedback/feedback.jsonl", "r") as f:
            lines = f.readlines()
        
        if not lines:
            return "📝 No feedback received yet."
        
        # Get last 10 feedback entries
        recent_feedback = lines[-10:]
        
        output = "### 📝 Recent Feedback (Last 10 entries)\n\n"
        for idx, line in enumerate(reversed(recent_feedback), 1):
            try:
                feedback = json.loads(line)
                emoji = "👍" if feedback['feedback'] == 'positive' else "👎"
                timestamp = feedback.get('timestamp', 'N/A')
                session = feedback.get('session_id', 'Unknown')[:12]
                
                output += f"**{idx}.** {emoji} | Session: `{session}` | Time: {timestamp}\n"
                if feedback.get('comment'):
                    output += f"   Comment: {feedback['comment']}\n"
                output += "\n"
            except:
                continue
        
        return output
    
    except Exception as e:
        return f"❌ Error reading feedback: {str(e)}"

def clear_all_memory(password: str) -> str:
    """Clear all conversation memories"""
    if not check_password(password):
        return "❌ **Incorrect Password!** Access denied."
    
    # This would require a new endpoint in the backend
    # For now, just show a message
    return "⚠️ This feature requires backend endpoint implementation. Manual restart of backend will clear all sessions."

# Create Admin Panel
with gr.Blocks(
    theme=gr.themes.Base(primary_hue="red"),
    title="College Chatbot - Admin Panel",
    css="""
    .gradio-container {max-width: 1000px !important;}
    footer {display: none !important;}
    """
) as admin:
    
    # Header
    gr.Markdown("""
    # 🔐 College Chatbot - Admin Panel
    ### Manage documents, view analytics, and monitor system health
    
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
        
        # Tab 1: Upload Documents
        with gr.Tab("📤 Upload Documents"):
            gr.Markdown("""
            ### Upload College Documents
            Upload PDFs (prospectus, syllabus) or CSVs (fees, courses) to update the knowledge base.
            
            **Supported Formats:**
            - **PDF**: College prospectus, syllabus, admission guidelines
            - **CSV**: Fee structure, course details, eligibility criteria
            """)
            
            file_upload = gr.File(
                label="Select Files to Upload",
                file_count="multiple",
                file_types=[".pdf", ".csv"],
                type="filepath"
            )
            
            upload_btn = gr.Button("🚀 Upload Files", variant="primary", size="lg")
            upload_output = gr.Markdown()
            
            upload_btn.click(
                upload_files,
                inputs=[password_input, file_upload],
                outputs=[upload_output]
            )
            
            gr.Markdown("""
            ### 💡 Tips
            - Upload multiple files at once for batch processing
            - Tables in PDFs are automatically extracted
            - CSV files are processed row-by-row with metadata
            - System uses smart chunking to preserve context
            - Changes are immediately reflected in student chatbot
            """)
        
        # Tab 2: Statistics & Analytics
        with gr.Tab("📊 Statistics"):
            gr.Markdown("### System Statistics and Analytics")
            
            stats_output = gr.Markdown()
            refresh_stats_btn = gr.Button("🔄 Refresh Statistics", variant="primary")
            
            refresh_stats_btn.click(
                get_statistics,
                inputs=[password_input],
                outputs=[stats_output]
            )
            
            # Load stats on tab open
            admin.load(get_statistics, inputs=[password_input], outputs=[stats_output])
        
        # Tab 3: Feedback Management
        with gr.Tab("📝 Feedback"):
            gr.Markdown("### User Feedback & Reviews")
            
            feedback_output = gr.Markdown()
            refresh_feedback_btn = gr.Button("🔄 Load Recent Feedback", variant="primary")
            
            refresh_feedback_btn.click(
                view_feedback,
                inputs=[password_input],
                outputs=[feedback_output]
            )
        
        # Tab 4: System Management
        with gr.Tab("⚙️ System"):
            gr.Markdown("""
            ### System Management
            Advanced operations for system maintenance
            """)
            
            gr.Markdown("#### 🗑️ Clear All Conversations")
            gr.Markdown("Remove all active conversation histories. Students will need to start fresh conversations.")
            
            clear_memory_btn = gr.Button("Clear All Memories", variant="stop")
            clear_output = gr.Markdown()
            
            clear_memory_btn.click(
                clear_all_memory,
                inputs=[password_input],
                outputs=[clear_output]
            )
            
            gr.Markdown("---")
            
            gr.Markdown("""
            #### 🔧 Backend Information
            - **API Endpoint:** `http://localhost:8000`
            - **API Documentation:** http://localhost:8000/docs
            - **Health Check:** http://localhost:8000/health
            
            #### 📚 System Features
            - ✅ Query Rewriting with LLMChain
            - ✅ Hybrid Search (Vector 70% + BM25 30%)
            - ✅ Cross-Encoder Re-ranking
            - ✅ Context Compression
            - ✅ Conversation Memory
            - ✅ Smart Chunking
            - ✅ Citation Support
            """)
        
        # Tab 5: Help
        with gr.Tab("❓ Help"):
            gr.Markdown("""
            ## 📖 Admin Panel Guide
            
            ### 🔐 Security
            - **Default Password:** `admin123`
            - **⚠️ IMPORTANT:** Change the password in `admin_panel.py` before deployment!
            - Location: Line 15 `ADMIN_PASSWORD = "admin123"`
            
            ### 📤 Uploading Documents
            
            **Step 1:** Enter your admin password  
            **Step 2:** Go to "Upload Documents" tab  
            **Step 3:** Click "Select Files" and choose PDFs or CSVs  
            **Step 4:** Click "Upload Files"  
            **Step 5:** Wait for success confirmation  
            
            **Best Practices:**
            - Name files clearly (e.g., `fees_2025.csv`, `prospectus_2025.pdf`)
            - Keep CSVs well-structured with clear headers
            - Upload updated documents to refresh information
            - Test queries after uploading
            
            ### 📊 Monitoring Performance
            
            **Key Metrics to Watch:**
            - **Satisfaction Rate:** Should be > 80%
            - **Total Documents:** Ensure all required docs are uploaded
            - **Active Conversations:** Monitor user engagement
            - **Feedback:** Review negative feedback for improvements
            
            ### 🐛 Troubleshooting
            
            **Problem:** Upload fails  
            **Solution:** Check backend is running (`python main.py`)
            
            **Problem:** Statistics not loading  
            **Solution:** Verify password and backend connection
            
            **Problem:** Students getting wrong answers  
            **Solution:** Upload more comprehensive documents or update existing ones
            
            ### 📞 Support
            - Backend Logs: Check terminal where `python main.py` is running
            - API Docs: http://localhost:8000/docs
            - Test Endpoint: http://localhost:8000/health
            """)
    
    # Footer
    gr.Markdown("""
    ---
    🔒 **Security Notice:** This panel should only be accessible to authorized college administrators.
    Change the default password before deploying to production.
    
    🔗 **Related Services:**
    - Student Chatbot: http://localhost:7860 (Gradio) or http://localhost:8501 (Streamlit)
    - Backend API: http://localhost:8000
    """)

if __name__ == "__main__":
    print("🔐 Starting Admin Panel...")
    print("🌐 Opening at: http://localhost:7861")
    print("🔑 Default Password: admin123 (CHANGE THIS!)")
    print("\n⚠️  Make sure backend is running: python main.py\n")
    
    admin.launch(
        server_name="0.0.0.0",
        server_port=7861,  # Different port from student interface
        share=True,
        show_error=True,
        auth=None  # Can add: auth=("admin", "password") for basic auth
    )