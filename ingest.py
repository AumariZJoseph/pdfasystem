import os
import faiss
from llama_index.core import SimpleDirectoryReader, VectorStoreIndex, StorageContext
from llama_index.core.node_parser import SentenceSplitter
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.vector_stores.faiss import FaissVectorStore

def ingest_data():
    # Validate data directory exists
    data_dir = "data"
    if not os.path.exists(data_dir) or not os.listdir(data_dir):
        raise FileNotFoundError(f"No documents found in {data_dir} directory")

    # Load documents with filename as source metadata
    documents = SimpleDirectoryReader(
        data_dir,
        required_exts=[".pdf", ".txt"],
        file_metadata=lambda filename: {"source": os.path.basename(filename)}
    ).load_data()
    
    # Split documents into chunks
    splitter = SentenceSplitter(
        chunk_size=512,
        chunk_overlap=64,
        include_metadata=True  # Ensure metadata is preserved
    )
    nodes = splitter.get_nodes_from_documents(documents)
    
    # Ensure source metadata exists for every chunk
    for node in nodes:
        # Extract filename only from source path
        source = node.metadata.get("source", "Unknown")
        if os.path.sep in source:
            source = os.path.basename(source)
        node.metadata["source"] = source
    
    # Create embeddings and vector store
    embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-small-en-v1.5")
    dimension = 384  # Dimension for bge-small-en-v1.5 embeddings
    
    # Initialize FAISS index
    faiss_index = faiss.IndexFlatL2(dimension)
    vector_store = FaissVectorStore(faiss_index)
    
    # Create storage context
    storage_context = StorageContext.from_defaults(
        vector_store=vector_store,
        persist_dir="storage"
    )
    
    # Create index with metadata
    index = VectorStoreIndex(
        nodes, 
        embed_model=embed_model, 
        storage_context=storage_context,
        show_progress=True  # Visual feedback during processing
    )
    
    # Persist index to storage directory
    index.storage_context.persist(persist_dir="storage")
    return f"Processed {len(nodes)} chunks from {len(documents)} documents"

if __name__ == "__main__":
    try:
        result = ingest_data()
        print(f"✅ {result}")
        print("Data ingestion complete with source metadata!")
    except Exception as e:
        print(f"❌ Ingestion failed: {str(e)}")
