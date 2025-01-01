from rag.vector_store import LegalVectorStore
from agents.legal_agent import LegalResearchAgent
from models.fine_tuner import LegalModelFineTuner
from utils.vertex_ai_client import VertexAIManager
import streamlit as st

class LexibotApp:
    def __init__(self):
        self.vector_store = LegalVectorStore()
        self.fine_tuner = LegalModelFineTuner()
        self.agent = LegalResearchAgent(self.vector_store, self.fine_tuner.model)
    
    def run_streamlit_app(self):
        st.title("Lexibot - Legal AI Assistant")
        
        query = st.text_input("Enter your legal question:")
        
        if st.button("Search") and query:
            with st.spinner("Processing..."):
                response = self.agent.process_query(query)
                st.write(response)
    
    def add_sample_data(self):
        sample_docs = [
            "Legal precedent regarding contract law...",
            "Constitutional amendment interpretation...",
            "Criminal law statute analysis..."
        ]
        self.vector_store.add_documents(sample_docs)

if __name__ == "__main__":
    app = LexibotApp()
    app.add_sample_data()
    app.run_streamlit_app()
