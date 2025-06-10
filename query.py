import os
from groq import Groq
from dotenv import load_dotenv
from llama_index.core import StorageContext, load_index_from_storage
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.vector_stores.faiss import FaissVectorStore
from llama_index.llms.groq import Groq as GroqLLM
from llama_index.core import PromptTemplate

load_dotenv()

class QASystem:
    def __init__(self):
        # Updated initialization
        self.llm = Groq(model="meta-llama/llama-4-scout-17b-16e-instruct", temperature=0.3)
        self.embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-small-en-v1.5")
        self.index = self._load_index()
        
    def _load_index(self):
        try:
            storage_context = StorageContext.from_defaults(
                persist_dir="storage",
                vector_store=FaissVectorStore.from_persist_dir("storage")
            )
            return load_index_from_storage(
                storage_context,
                embed_model=self.embed_model
            )
        except Exception as e:
            raise RuntimeError(f"Failed to load index: {str(e)}")
    
    def ask(self, question):
        qa_prompt = PromptTemplate(
            "Context information:\n{context_str}\n\n"
            "Question: {query_str}\n\n"
            "Answer clearly and concisely based on the context.\nAnswer:"
        )
        
        query_engine = self.index.as_query_engine(
            similarity_top_k=3,
            llm=self.llm,
            text_qa_template=qa_prompt,
            streaming=False
        )
        response = query_engine.query(question)
        return response.response

