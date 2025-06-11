import streamlit as st
import os
import shutil
import time
from ingest import ingest_data
from query import QASystem

# Configuration - use Streamlit's temp directories
DATA_DIR = "data"
STORAGE_DIR = "storage"

# Initialize session state
if 'qa_system' not in st.session_state:
    st.session_state.qa_system = None
if 'knowledge_base_built' not in st.session_state:
    st.session_state.knowledge_base_built = False
if 'uploaded_files' not in st.session_state:
    st.session_state.uploaded_files = []

# Create directories if missing
os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(STORAGE_DIR, exist_ok=True)

# --- Sidebar: File Management ---
st.sidebar.header("📂 File Management")

# Display existing files
st.sidebar.subheader("Current Documents")
for i, file in enumerate(st.session_state.uploaded_files):
    cols = st.sidebar.columns([4, 1])
    cols[0].write(f"📄 {file['name']}")
    if cols[1].button("🗑️", key=f"del_{i}"):
        # Remove file from session state and disk
        os.remove(os.path.join(DATA_DIR, file['name']))
        st.session_state.uploaded_files.pop(i)
        st.session_state.knowledge_base_built = False
        st.rerun()

# File upload
st.sidebar.subheader("Add Documents")
uploaded_file = st.sidebar.file_uploader(
    "Choose PDF/TXT file",
    type=["pdf", "txt"],
    label_visibility="collapsed"
)

if uploaded_file:
    save_path = os.path.join(DATA_DIR, uploaded_file.name)
    with open(save_path, "wb") as f:
        f.write(uploaded_file.getbuffer())
    
    # Add to session state
    st.session_state.uploaded_files.append({
        "name": uploaded_file.name,
        "size": uploaded_file.size,
        "type": uploaded_file.type
    })
    st.session_state.knowledge_base_built = False
    st.rerun()

# --- Knowledge Base Management ---
st.sidebar.header("🛠️ Knowledge Base")

# Rebuild button
if st.sidebar.button("🔨 Rebuild Knowledge Base", 
                    disabled=len(st.session_state.uploaded_files) == 0,
                    help="Process current documents"):
    with st.sidebar.status("📚 Analyzing documents...", expanded=True) as status:
        try:
            # Clear existing storage
            if os.path.exists(STORAGE_DIR):
                shutil.rmtree(STORAGE_DIR)
                st.write("Cleared previous knowledge base")
            
            # Ingest new data
            st.write("Processing documents...")
            ingest_data()
            
            # Initialize QA system
            st.write("Initializing QA system...")
            st.session_state.qa_system = QASystem()
            st.session_state.knowledge_base_built = True
            
            status.update(label="✅ Knowledge base updated!", state="complete")
            st.balloons()
        except Exception as e:
            status.update(label=f"❌ Error: {str(e)}", state="error")

# System status
st.sidebar.divider()
st.sidebar.subheader("System Status")
if st.session_state.uploaded_files:
    st.sidebar.info(f"📄 {len(st.session_state.uploaded_files)} documents uploaded")
else:
    st.sidebar.warning("⚠️ No documents uploaded")
    
if st.session_state.knowledge_base_built:
    st.sidebar.success("✅ Knowledge base ready")
else:
    st.sidebar.warning("⚠️ Knowledge base not built")

# --- Main Interface ---
st.title("📄 Document QA Assistant")

# Chat interface
if prompt := st.chat_input("Ask about your documents..."):
    if not st.session_state.uploaded_files:
        st.error("⚠️ Please upload documents first!")
    elif not st.session_state.knowledge_base_built:
        st.error("⚠️ Please rebuild knowledge base first!")
    else:
        # Display user message
        with st.chat_message("user"):
            st.write(prompt)
        
        # Get response
        with st.chat_message("assistant"):
            with st.spinner("🤔 Thinking..."):
                try:
                    response = st.session_state.qa_system.ask(prompt)
                    st.write(response)
                except Exception as e:
                    st.error(f"⚠️ Error: {str(e)}")
