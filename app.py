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

qa = init_qa_system()

# --- Sidebar: File Management ---
st.sidebar.header("📂 File Management")

# Display existing files with delete buttons
st.sidebar.subheader("Current Documents")
existing_files = os.listdir(DATA_DIR)
for file in existing_files:
    cols = st.sidebar.columns([4,1])
    cols[0].write(f"📄 {file}")
    if cols[1].button("🗑️", key=f"del_{file}"):
        os.remove(os.path.join(DATA_DIR, file))
        st.rerun()

# File upload section with proper state reset
st.sidebar.subheader("Add Documents")

# Use a separate key to track uploader state
if 'uploader_key' not in st.session_state:
    st.session_state.uploader_key = 0

uploaded_file = st.sidebar.file_uploader(
    "Choose PDF/TXT file",
    type=["pdf", "txt"],
    label_visibility="collapsed",
    key=f"file_uploader_{st.session_state.uploader_key}"
)

if uploaded_file:
    save_path = os.path.join(DATA_DIR, uploaded_file.name)
    with open(save_path, "wb") as f:
        f.write(uploaded_file.getbuffer())
    # Increment key to reset uploader
    st.session_state.uploader_key += 1
    st.rerun()

# --- Always Visible Rebuild Button ---
st.sidebar.header("🛠️ Knowledge Base")

# Get current file state
has_files = len(os.listdir(DATA_DIR)) > 0

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
            
            # Clear cache and re-ingest
            init_qa_system.clear()
            ingest_data()
            
            st.sidebar.success("✅ Knowledge base updated!")
            st.balloons()
        except Exception as e:
            st.sidebar.error(f"❌ Error: {str(e)}")

# --- Main Interface ---
st.title("📄 Document QA Assistant")

# Status indicators
if not has_files:
    st.warning("No documents uploaded. Add files using the sidebar. ➡️")
elif not os.path.exists(STORAGE_DIR):
    st.success("✅ Documents ready! Click 'Rebuild Knowledge Base' to process")

# Chat interface
question = st.chat_input("Ask about your documents...")
if question:
    if not has_files:
        st.error("⚠️ Please upload documents first!")
    elif not os.path.exists(STORAGE_DIR):
        st.error("⚠️ Please rebuild knowledge base first!")
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