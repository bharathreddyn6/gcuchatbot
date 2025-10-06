"""
College Chatbot - Student Interface (Streamlit)
Clean chat interface without upload options - for end users
"""

import streamlit as st
import requests
from datetime import datetime
import uuid

# Page configuration
st.set_page_config(
    page_title="College Information Assistant",
    page_icon="🎓",
    layout="centered",
    initial_sidebar_state="collapsed"
)

# API Configuration
API_BASE = "http://localhost:8000"

# Initialize session state
if 'session_id' not in st.session_state:
    st.session_state.session_id = f"student_{uuid.uuid4().hex[:8]}"

if 'messages' not in st.session_state:
    st.session_state.messages = []

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        padding: 1rem 0;
    }
    .sub-header {
        text-align: center;
        color: #666;
        margin-bottom: 2rem;
        font-size: 1.1rem;
    }
    .contact-box {
        background-color: #f0f8ff;
        border-left: 4px solid #1f77b4;
        padding: 1rem;
        margin-top: 2rem;
        border-radius: 5px;
    }
    /* Hide Streamlit branding */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
</style>
""", unsafe_allow_html=True)

# Header
st.markdown('<div class="main-header">🎓 College Information Assistant</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-header">Ask me anything about courses, fees, eligibility, and admissions!</div>', unsafe_allow_html=True)

# Display chat messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        
        # Display sources if available
        if "sources" in message and message["sources"]:
            with st.expander("📚 View Sources"):
                for source in message["sources"]:
                    st.markdown(f"• {source}")

# Chat input
if prompt := st.chat_input("Type your question here..."):
    # Add user message
    st.session_state.messages.append({"role": "user", "content": prompt})
    
    with st.chat_message("user"):
        st.markdown(prompt)
    
    # Get bot response
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            try:
                response = requests.post(
                    f"{API_BASE}/chat",
                    json={
                        "query": prompt,
                        "session_id": st.session_state.session_id
                    },
                    timeout=30
                )
                
                if response.status_code == 200:
                    result = response.json()
                    answer = result['answer']
                    sources = result.get('sources', [])
                    
                    st.markdown(answer)
                    
                    # Display sources
                    if sources:
                        with st.expander("📚 View Sources"):
                            for source in sources[:5]:
                                st.markdown(f"• {source}")
                    
                    # Store in session
                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": answer,
                        "sources": sources
                    })
                    
                    # Feedback buttons
                    col1, col2, col3 = st.columns([1, 1, 8])
                    with col1:
                        if st.button("👍", key=f"up_{len(st.session_state.messages)}"):
                            requests.post(
                                f"{API_BASE}/feedback",
                                json={
                                    "message_id": f"msg_{datetime.now().timestamp()}",
                                    "feedback": "positive",
                                    "session_id": st.session_state.session_id
                                }
                            )
                            st.success("Thanks!")
                    with col2:
                        if st.button("👎", key=f"down_{len(st.session_state.messages)}"):
                            requests.post(
                                f"{API_BASE}/feedback",
                                json={
                                    "message_id": f"msg_{datetime.now().timestamp()}",
                                    "feedback": "negative",
                                    "session_id": st.session_state.session_id
                                }
                            )
                            st.info("Feedback noted")
                else:
                    error_msg = "❌ Sorry, I'm having trouble answering right now. Please try again."
                    st.error(error_msg)
                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": error_msg
                    })
            
            except Exception as e:
                error_msg = "❌ Unable to connect. Please ensure the backend is running."
                st.error(error_msg)
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": error_msg
                })

# Example questions
if len(st.session_state.messages) == 0:
    st.divider()
    st.markdown("### 💡 Popular Questions")
    
    col1, col2 = st.columns(2)
    
    with col1:
        if st.button("🎓 Which branches are available?", use_container_width=True):
            st.session_state.messages.append({"role": "user", "content": "Which branches are available?"})
            st.rerun()
        
        if st.button("💰 What is the fee for CSE?", use_container_width=True):
            st.session_state.messages.append({"role": "user", "content": "What is the fee structure for CSE?"})
            st.rerun()
        
        if st.button("📋 Eligibility criteria?", use_container_width=True):
            st.session_state.messages.append({"role": "user", "content": "What are the eligibility criteria for BTech?"})
            st.rerun()
        
        if st.button("🪑 How many seats available?", use_container_width=True):
            st.session_state.messages.append({"role": "user", "content": "How many seats are available in each branch?"})
            st.rerun()
    
    with col2:
        if st.button("🔐 About Cybersecurity specialization?", use_container_width=True):
            st.session_state.messages.append({"role": "user", "content": "Tell me about the Cybersecurity specialization"})
            st.rerun()
        
        if st.button("📊 Lateral entry process?", use_container_width=True):
            st.session_state.messages.append({"role": "user", "content": "Can I do lateral entry? What are the requirements?"})
            st.rerun()
        
        if st.button("📝 Admission process?", use_container_width=True):
            st.session_state.messages.append({"role": "user", "content": "What is the admission process?"})
            st.rerun()
        
        if st.button("⭐ NAAC accreditation?", use_container_width=True):
            st.session_state.messages.append({"role": "user", "content": "Is the college NAAC accredited?"})
            st.rerun()

# Footer with contact info
st.markdown("""
<div class="contact-box">
    <h4>💬 Need More Help?</h4>
    <p><strong>Contact our Admissions Office:</strong></p>
    <ul>
        <li>📧 <strong>Email:</strong> admissions@college.edu</li>
        <li>📞 <strong>Phone:</strong> +91 12345 67890</li>
        <li>🌐 <strong>Website:</strong> www.college.edu</li>
        <li>📍 <strong>Visit:</strong> Campus Address, City, State</li>
    </ul>
    <p><em>This chatbot uses AI to provide information. For official details, please contact the admissions office.</em></p>
</div>
""", unsafe_allow_html=True)

# Clear chat button (subtle, at bottom)
if st.session_state.messages:
    if st.button("🗑️ Start New Conversation", use_container_width=True):
        try:
            requests.delete(f"{API_BASE}/clear_memory/{st.session_state.session_id}")
        except:
            pass
        st.session_state.messages = []
        st.rerun()