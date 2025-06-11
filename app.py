import streamlit as st
import os
import shutil
from ingest import ingest_data
from query import QASystem

# Configuration
DATA_DIR = "data"
STORAGE_DIR = "storage"

# Create directories if missing
os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(STORAGE_DIR, exist_ok=True)

# Initialize QA System
@st.cache_resource
def init_qa_system():
    return QASystem()

# Only initialize QA system if storage exists
if os.path.exists(STORAGE_DIR) and os.listdir(STORAGE_DIR):
    qa = init_qa_system()
else:
    qa = None

# --- Sidebar: File Management ---
st.sidebar.header("📂 File Management")

# Use session state for file tracking
if 'file_list' not in st.session_state:
    st.session_state.file_list = os.listdir(DATA_DIR)

# Display existing files with delete buttons
st.sidebar.subheader("Current Documents")
for file in st.session_state.file_list[:]:  # Iterate over copy
    cols = st.sidebar.columns([4,1])
    cols[0].write(f"📄 {file}")
    if cols[1].button("🗑️", key=f"del_{file}"):
        try:
            os.remove(os.path.join(DATA_DIR, file))
            st.session_state.file_list.remove(file)
            st.rerun()
        except Exception as e:
            st.sidebar.error(f"Delete failed: {str(e)}")

# File upload section
st.sidebar.subheader("Add Documents")
uploaded_file = st.sidebar.file_uploader(
    "Choose PDF/TXT file",
    type=["pdf", "txt"],
    label_visibility="collapsed",
    key="file_uploader"
)

if uploaded_file is not None:
    # Only process if it's a new file
    if uploaded_file.name not in st.session_state.file_list:
        save_path = os.path.join(DATA_DIR, uploaded_file.name)
        with open(save_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
        st.session_state.file_list.append(uploaded_file.name)
        st.rerun()

# --- Always Visible Rebuild Button ---
st.sidebar.header("🛠️ Knowledge Base")

# Get current file state
has_files = len(st.session_state.file_list) > 0

# Rebuild button with force-enable
if st.sidebar.button(
    "🔨 Rebuild Knowledge Base",
    disabled=not has_files,
    help="Process current documents" if has_files else "Upload files first"
):
    with st.spinner("📚 Analyzing documents..."):
        try:
            # Clear existing storage
            if os.path.exists(STORAGE_DIR):
                shutil.rmtree(STORAGE_DIR)
                os.makedirs(STORAGE_DIR, exist_ok=True)
            
            # Clear cache and re-ingest
            init_qa_system.clear()
            ingest_data()
            
            # Reinitialize QA system
            qa = init_qa_system()
            
            st.sidebar.success("✅ Knowledge base updated!")
            st.balloons()
        except Exception as e:
            st.sidebar.error(f"❌ Error: {str(e)}")

# --- Main Interface ---
st.title("📄 Document QA Assistant")

# Status indicators
if not has_files:
    st.warning("No documents uploaded. Add files using the sidebar. ➡️")
elif not os.path.exists(STORAGE_DIR) or not os.listdir(STORAGE_DIR):
    st.success("✅ Documents ready! Click 'Rebuild Knowledge Base' to process")

# Chat interface
question = st.chat_input("Ask about your documents...")
if question:
    if not has_files:
        st.error("⚠️ Please upload documents first!")
    elif not os.path.exists(STORAGE_DIR) or not os.listdir(STORAGE_DIR):
        st.error("⚠️ Please rebuild knowledge base first!")
    elif qa is None:
        st.error("⚠️ System not ready - please rebuild knowledge base")
    else:
        with st.chat_message("user"):
            st.write(question)
        
        with st.chat_message("assistant"):
            with st.spinner("🤔 Thinking..."):
                try:
                    answer = qa.ask(question)
                    st.write(answer)
                except Exception as e:
                    st.error(f"⚠️ Error: {str(e)}")
