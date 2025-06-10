import os
from dotenv import load_dotenv
from llama_index.core import StorageContext, VectorStoreIndex
from llama_index.vector_stores.faiss import FaissVectorStore
from llama_index.llms.groq import Groq as GroqLLM
from llama_index.core import PromptTemplate

load_dotenv()

class QASystem:
    def __init__(self):
        self.llm = GroqLLM(model="llama3-70b-8192", temperature=0.3)
        self.index = self._load_index()
        
    def _load_index(self):
        storage_context = StorageContext.from_defaults(
            persist_dir="storage",
            vector_store=FaissVectorStore.from_persist_dir("storage")
        )
        return VectorStoreIndex.from_vector_store(
            storage_context.vector_store, 
            storage_context=storage_context
        )
    
    def ask(self, question):
        # Custom prompt with source awareness
        qa_prompt = PromptTemplate(
            "You are an expert document analyst. Given information from multiple sources:\n"
            "----------------\n"
            "{context_str}\n"
            "----------------\n"
            "Answer the question: {query_str}\n"
            "Structure your response with:\n"
            "1. A concise main answer\n"
            "2. Source attribution using the document names\n"
            "3. Key supporting evidence\n"
            "Answer:"
        )
        
        # Retrieve relevant context with metadata
        retriever = self.index.as_retriever(similarity_top_k=4)
        context_nodes = retriever.retrieve(question)
        
        # Format context with source markers
        context_str = ""
        for i, node in enumerate(context_nodes):
            source = node.metadata.get("source", "Unknown Document")
            context_str += f"[Document: {source}]\n{node.text}\n\n"
        
        # Generate answer with source attribution
        llm_response = self.llm.complete(
            qa_prompt.format(context_str=context_str, query_str=question)  # Fixed parenthesis here
        
        # Extract and format sources
        sources = {node.metadata.get("source", "Unknown Document") 
                  for node in context_nodes}
        source_list = "\n".join([f"• {source}" for source in sources])
        
        # Final formatted response
        return f"{llm_response.text}\n\n🔍 Source Documents:\n{source_list}"

def main():  # Properly indented function
    qa = QASystem()
    print("Document QA System - Type 'exit' to quit\n")
    
    while True:
        question = input("\nQuestion: ")
        if question.lower() in ['exit', 'quit', 'q']:
            break
            
        print("\nProcessing...")
        try:
            answer = qa.ask(question)
            print(f"\nAnswer:\n{answer}")
        except Exception as e:
            print(f"\n⚠️ Error: {str(e)}")

if __name__ == "__main__":
    main()  # Correctly placed
