import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from main import LexibotApp

if __name__ == "__main__":
    app = LexibotApp()
    app.add_sample_data()
    print("Lexibot Legal AI Assistant initialized!")
    
    # Demo query
    response = app.agent.process_query("What are the key principles of contract law?")
    print(f"Demo Response: {response}")
