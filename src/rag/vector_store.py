import chromadb
from sentence_transformers import SentenceTransformer
from typing import List, Dict

class LegalVectorStore:
    def __init__(self, collection_name: str = "legal_docs"):
        self.client = chromadb.Client()
        self.collection = self.client.get_or_create_collection(collection_name)
        self.encoder = SentenceTransformer('all-MiniLM-L6-v2')
    
    def add_documents(self, documents: List[str], metadatas: List[Dict] = None):
        embeddings = self.encoder.encode(documents)
        ids = [f"doc_{i}" for i in range(len(documents))]
        self.collection.add(
            embeddings=embeddings.tolist(),
            documents=documents,
            metadatas=metadatas or [{}] * len(documents),
            ids=ids
        )
    
    def search(self, query: str, n_results: int = 5):
        query_embedding = self.encoder.encode([query])
        results = self.collection.query(
            query_embeddings=query_embedding.tolist(),
            n_results=n_results
        )
        return results['documents'][0]
