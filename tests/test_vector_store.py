import unittest
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from rag.vector_store import LegalVectorStore

class TestVectorStore(unittest.TestCase):
    def setUp(self):
        self.store = LegalVectorStore("test_collection")
    
    def test_add_and_search(self):
        docs = ["Contract law basics", "Criminal procedure"]
        self.store.add_documents(docs)
        results = self.store.search("contract", n_results=1)
        self.assertTrue(len(results) > 0)

if __name__ == '__main__':
    unittest.main()
