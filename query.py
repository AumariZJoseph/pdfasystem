import os
from groq import Groq as GroqClient
from dotenv import load_dotenv
from llama_index.core import StorageContext, load_index_from_storage
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.vector_stores.faiss import FaissVectorStore
from llama_index.llms.groq import Groq  # Updated import
from llama_index.core import PromptTemplate
import streamlit as st

load_dotenv()

class QASystem:
    def __init__(self):
        # Use Groq instead of GroqLLM
        self.llm = Groq(model="meta-llama/llama-4-scout-17b-16e-instruct", temperature=0.3)
        self.embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-small-en-v1.5")
        self.index = None
        self._load_index()
        
    def _load_index(self):
        """Load index from storage with error handling"""
        try:
            if not os.path.exists("storage") or not os.listdir("storage"):
                raise ValueError("Storage directory is empty or missing")
                
            storage_context = StorageContext.from_defaults(
                persist_dir="storage",
                vector_store=FaissVectorStore.from_persist_dir("storage")
            )
            self.index = load_index_from_storage(
                storage_context,
                embed_model=self.embed_model
            )
        except Exception as e:
            st.error(f"⚠️ Failed to load index: {str(e)}")
            self.index = None
    
    def ask(self, question):
        if self.index is None:
            return "⚠️ System not ready - please rebuild knowledge base"
            
        qa_prompt = PromptTemplate(
            "Context information from multiple sources:\n"
            "----------------\n"
            "{context_str}\n"
            "----------------\n"
            "Given this information, answer the question: {query_str}\n"
            "Structure your answer as:\n"
            "1. Main answer\n"
            "2. Sources (list document names)\n"
            "Answer:"
        )
        
        # Retrieve with metadata
        retriever = self.index.as_retriever(similarity_top_k=3)
        context_nodes = retriever.retrieve(question)
        
        # Format context with sources
        context_str = ""
        for i, node in enumerate(context_nodes):
            source = node.metadata.get("source", "Unknown Document")
            context_str += f"[Source {i+1}: {source}]\n{node.text}\n\n"
        
        # Generate answer with source attribution
        response = self.llm.complete(
            qa_prompt.format(context_str=context_str, query_str=question)
        )
        
        # Extract sources
        sources = {node.metadata.get("source", "Unknown") for node in context_nodes}
        source_list = "\n- " + "\n- ".join(sources)
        
        return f"{response.text}\n\nSources:{source_list}"
