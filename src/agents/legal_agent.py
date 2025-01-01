from langgraph.graph import Graph
from langchain.schema import HumanMessage
from typing import Dict, Any
import json

class LegalResearchAgent:
    def __init__(self, vector_store, llm):
        self.vector_store = vector_store
        self.llm = llm
        self.graph = self._create_workflow()
    
    def _create_workflow(self):
        graph = Graph()
        
        graph.add_node("query_analysis", self._analyze_query)
        graph.add_node("document_retrieval", self._retrieve_documents)
        graph.add_node("legal_reasoning", self._legal_reasoning)
        graph.add_node("citation_generation", self._generate_citations)
        
        graph.add_edge("query_analysis", "document_retrieval")
        graph.add_edge("document_retrieval", "legal_reasoning")
        graph.add_edge("legal_reasoning", "citation_generation")
        
        graph.set_entry_point("query_analysis")
        return graph.compile()
    
    def _analyze_query(self, state: Dict[str, Any]) -> Dict[str, Any]:
        query = state["query"]
        # Simple query analysis
        state["analyzed_query"] = {
            "intent": "legal_research",
            "entities": [],
            "query": query
        }
        return state
    
    def _retrieve_documents(self, state: Dict[str, Any]) -> Dict[str, Any]:
        query = state["analyzed_query"]["query"]
        docs = self.vector_store.search(query, n_results=5)
        state["retrieved_docs"] = docs
        return state
    
    def _legal_reasoning(self, state: Dict[str, Any]) -> Dict[str, Any]:
        docs = state["retrieved_docs"]
        context = "\n".join(docs[:3])
        state["reasoning"] = f"Based on retrieved documents: {context[:200]}..."
        return state
    
    def _generate_citations(self, state: Dict[str, Any]) -> Dict[str, Any]:
        state["citations"] = ["Legal Doc 1", "Legal Doc 2"]
        state["final_answer"] = f"{state['reasoning']}\n\nCitations: {', '.join(state['citations'])}"
        return state
    
    def process_query(self, query: str) -> str:
        initial_state = {"query": query}
        result = self.graph.invoke(initial_state)
        return result["final_answer"]
