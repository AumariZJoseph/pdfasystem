import os
import faiss
from llama_index.core import SimpleDirectoryReader, VectorStoreIndex, StorageContext
from llama_index.core.node_parser import SentenceSplitter
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.vector_stores.faiss import FaissVectorStore

def ingest_data():
    # Streamlit-friendly document processing
    documents = []
    for filename in os.listdir("data"):
        if filename.endswith((".pdf", ".txt")):
            file_path = os.path.join("data", filename)
            loader = SimpleDirectoryReader(input_files=[file_path])
            docs = loader.load_data()
            for doc in docs:
                doc.metadata = {"source": filename}
            documents.extend(docs)
    
    # Process in chunks to manage memory
    splitter = SentenceSplitter(chunk_size=512, chunk_overlap=64)
    all_nodes = []
    
    for doc in documents:
        nodes = splitter.get_nodes_from_documents([doc])
        for node in nodes:
            node.metadata = doc.metadata
        all_nodes.extend(nodes)
    
    # Create vector store
    embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-small-en-v1.5")
    dimension = 384
    faiss_index = faiss.IndexFlatL2(dimension)
    vector_store = FaissVectorStore(faiss_index)
    storage_context = StorageContext.from_defaults(vector_store=vector_store)

    # Create and persist index
    index = VectorStoreIndex(
        all_nodes, 
        embed_model=embed_model, 
        storage_context=storage_context
    )
    index.storage_context.persist(persist_dir="storage")


