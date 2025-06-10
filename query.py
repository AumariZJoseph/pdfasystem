import os
from dotenv import load_dotenv
from llama_index.core import StorageContext, load_index_from_storage
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.vector_stores.faiss import FaissVectorStore
from llama_index.llms.groq import Groq 
from llama_index.core import PromptTemplate

load_dotenv()

class QASystem:
    def __init__(self):
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
        
        retriever = self.index.as_retriever(similarity_top_k=3)
        context_nodes = retriever.retrieve(question)
        
        context_str = ""
        for i, node in enumerate(context_nodes):
            source = node.metadata.get("source", "Unknown Document")
            context_str += f"[Source {i+1}: {source}]\n{node.text}\n\n"
        
        response = self.llm.complete(
            qa_prompt.format(context_str=context_str, query_str=question)
        )
        
        sources = {node.metadata.get("source", "Unknown") for node in context_nodes}
        source_list = "\n- " + "\n- ".join(sources)
        
        return f"{response.text}\n\nSources:{source_list}"
