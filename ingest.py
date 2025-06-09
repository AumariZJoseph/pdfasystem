import os
import faiss
from llama_index.core import SimpleDirectoryReader, VectorStoreIndex, StorageContext
from llama_index.core.node_parser import SentenceSplitter
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.vector_stores.faiss import FaissVectorStore

def ingest_data():
    # Load documents
    documents = SimpleDirectoryReader("data", required_exts=[".pdf", ".txt"]).load_data()
    
    # Split documents into chunks
    splitter = SentenceSplitter(chunk_size=512, chunk_overlap=64)
    nodes = splitter.get_nodes_from_documents(documents)
    
    # Create embeddings and vector store
    embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-small-en-v1.5")
    
    # Create FAISS index with correct dimensions
    dimension = 384  # bge-small-en-v1.5 uses 384-dimensional embeddings
    faiss_index = faiss.IndexFlatL2(dimension)
    vector_store = FaissVectorStore(faiss_index=faiss_index)
    
    storage_context = StorageContext.from_defaults(vector_store=vector_store)
    
    # Create and save index
    index = VectorStoreIndex(
        nodes, embed_model=embed_model, storage_context=storage_context
    )
    index.storage_context.persist(persist_dir="storage")

if __name__ == "__main__":
    ingest_data()
    print("Data ingestion complete!")