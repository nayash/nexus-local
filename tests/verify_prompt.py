import sys
import os

# Add src to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.core.organizer import Organizer

def test_prompt():
    print("Initializing Organizer...")
    org = Organizer()
    
    test_cases = [
        {
            "filename": "nexus-local-logs-debug.txt",
            "content": "2024-05-20 10:00:00 INFO: Starting application...\nERROR: Connection failed.",
            "expected": "Logs"
        },
        {
            "filename": "file-watcher-instruction.txt",
            "content": "Instructions for using the file watcher.\n1. Add folder.\n2. Wait.",
            "expected": "Instructions"
        },
        {
            "filename": "190217091-BI.pdf",
            "content": "INSURANCE POLICY DECLARATION\nPolicy Number: 123456789\nInsured: John Doe\nCoverage: Fire, Theft.",
            "expected": "InsuranceDocs"
        },
        {
            "filename": "NestedLearning.pdf",
            "content": "Abstract\nWe present a new method for nested learning in neural networks.\nIntroduction...",
            "expected": "ResearchPapers"
        }
    ]
    
    print("\n--- Testing New Prompt Logic ---\n")
    
    for case in test_cases:
        cat = org._categorize_file(case['filename'], case['content'], "None")
        print(f"File: {case['filename']}")
        print(f"Expected: {case['expected']}")
        print(f"Got:      {cat}")
        print("-" * 30)

if __name__ == "__main__":
    test_prompt()
