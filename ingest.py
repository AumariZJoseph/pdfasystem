import os
import faiss
from llama_index.core import SimpleDirectoryReader, VectorStoreIndex, StorageContext
from llama_index.core.node_parser import SentenceSplitter
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.vector_stores.faiss import FaissVectorStore
import streamlit as st

def ingest_data():
    """Process documents from data directory"""
    # Verify files exist
    if not os.path.exists("data") or len(os.listdir("data")) == 0:
        raise ValueError("No documents found in data directory")
    
    # Load documents with metadata
    documents = SimpleDirectoryReader(
        "data",
        required_exts=[".pdf", ".txt"],
        file_metadata=lambda filename: {"source": os.path.basename(filename)}
    ).load_data()
    
    # Split documents
    splitter = SentenceSplitter(chunk_size=512, chunk_overlap=64)
    nodes = splitter.get_nodes_from_documents(documents)
    
    # Add source metadata to each chunk
    for node in nodes:
        if "source" not in node.metadata:
            node.metadata["source"] = "Unknown"
    
    # Create embeddings and vector store
    embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-small-en-v1.5")
    dimension = 384
    faiss_index = faiss.IndexFlatL2(dimension)
    vector_store = FaissVectorStore(faiss_index)
    storage_context = StorageContext.from_defaults(vector_store=vector_store)
    
    # Create index with metadata
    index = VectorStoreIndex(
        nodes, 
        embed_model=embed_model, 
        storage_context=storage_context
    )
    index.storage_context.persist(persist_dir="storage")
