import os
import shutil
from src.rag.ingestion import NexusIngestor
from src.core.config import Config

def test_naive_ingestion_and_search():
    print("\n--- Testing Naive Strategy ---")
    ingestor = NexusIngestor(strategy="naive")
    
    # Create a dummy file
    test_file = "test_naive.txt"
    with open(test_file, "w") as f:
        f.write("This is a test document for naive ingestion. " * 50)
        
    try:
        # Ingest
        success, _, _ = ingestor.ingest_file(test_file, table_name="test_naive")
        assert success, "Naive ingestion failed"
        
        # Search
        results = ingestor.search("test document", k=1, table_name="test_naive")
        print(f"Search Results: {len(results)}")
        if results:
            print(f"Top Result Content: {results[0].page_content[:50]}...")
            
        assert len(results) > 0, "Naive search returned no results"
        
    finally:
        if os.path.exists(test_file):
            os.remove(test_file)

def test_parent_ingestion_and_search():
    print("\n--- Testing Parent Strategy ---")
    ingestor = NexusIngestor(strategy="parent")
    
    # Create a dummy file
    # Make it large enough to trigger parent splitting (>2000 chars)
    test_file = "test_parent.txt"
    content = "This is a parent chunk. " * 200 # ~4800 chars
    # Add a specific needle
    content += " NEEDLE_IN_HAYSTACK " + ("Filler text. " * 100)
    
    with open(test_file, "w") as f:
        f.write(content)
        
    try:
        # Ingest
        success, _, _ = ingestor.ingest_file(test_file)
        assert success, "Parent ingestion failed"
        
        # Search for needle
        results = ingestor.search("NEEDLE_IN_HAYSTACK", k=1)
        print(f"Search Results: {len(results)}")
        
        if results:
            doc = results[0]
            print(f"Top Result Content Length: {len(doc.page_content)}")
            print(f"Top Result Content Preview: {doc.page_content[:100]}...")
            
            # Check if it returned a large parent chunk
            # Child chunks are 400, Parent is 2000.
            # If length > 450, it's likely a parent.
            if len(doc.page_content) > 450:
                 print("✅ VERIFIED: Returned document is larger than child chunk size (Parent Document Retrieved).")
            else:
                 print("⚠️ WARNING: Returned document seems small. Might be a child or small parent.")
                 
        assert len(results) > 0, "Parent search returned no results"

    finally:
        if os.path.exists(test_file):
            os.remove(test_file)

if __name__ == "__main__":
    # Ensure dependencies are ready
    print("Starting tests...")
    test_naive_ingestion_and_search()
    test_parent_ingestion_and_search()
    print("\nAll tests passed!")
