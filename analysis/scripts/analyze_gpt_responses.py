#!/usr/bin/env python3
"""
Script to analyze GPT responses and understand the difference between
recommendations: [] and no recommendations field
"""

import json
import os
from pymongo import MongoClient

def analyze_gpt_responses():
    """Analyze GPT responses to understand the logic"""
    print("Analyzing GPT responses to understand recommendations logic...")
    
    # Load a sample batch result file
    batch_file = "gpt_batch_scripts/iteration_1/books/outputs/batch_result_books_batch_1.jsonl"
    
    if not os.path.exists(batch_file):
        print(f"[FAIL] Batch file not found: {batch_file}")
        return
    
    print(f"Analyzing: {batch_file}")
    
    with open(batch_file, 'r', encoding='utf-8') as f:
        for i, line in enumerate(f):
            if i >= 3:  # Analyze first 3 entries
                break
                
            if line.strip():
                try:
                    data = json.loads(line)
                    custom_id = data.get('custom_id', 'Unknown')
                    response_content = data.get('response', {}).get('body', {}).get('choices', [{}])[0].get('message', {}).get('content', '')
                    
                    print(f"\n{'='*60}")
                    print(f"ENTRY {i+1}: {custom_id}")
                    print(f"{'='*60}")
                    print(f"GPT Response:")
                    print(response_content)
                    print(f"\n{'='*60}")
                    
                    # Try to parse the response
                    try:
                        parsed = json.loads(response_content)
                        print(f"Parsed JSON:")
                        print(json.dumps(parsed, indent=2))
                    except:
                        print(f"Could not parse as JSON - raw response shown above")
                        
                except Exception as e:
                    print(f"Error processing line: {e}")

def analyze_extraction_logic():
    """Analyze the extraction logic from the scripts"""
    print("\nANALYZING EXTRACTION LOGIC:")
    print("="*60)
    
    print("""
KEY LOGIC FROM YOUR SCRIPTS:

1. GPT Response Processing:
   - GPT returns JSON with keys like 'c_comment_id': [recommendations]
   - If GPT finds entities: 'c_123': [{'title': 'Book', 'author': 'Author'}]
   - If GPT finds no entities: 'c_123': []  (empty array)
   - If GPT completely fails: No 'c_comment_id' key at all

2. Your Script Logic:
   ```python
   # Get all comment recommendations
   comment_recs = {}
   for key, items in reply_json.items():
       if key.startswith("c_") and isinstance(items, list):
           cm_id = key[2:]  # Remove 'c_' prefix
           comment_recs[cm_id] = items
   
   # Update database
   for cm_id, recs in comment_recs.items():
       collection.update_one(
           {"id": custom_id, "comments.id": cm_id},
           {"$set": {"comments.$.recommendations": recs}}
       )
   ```

3. The Key Difference:
   - If GPT returns 'c_123': [] → Your script sets recommendations: []
   - If GPT returns no 'c_123' key → Your script doesn't update that comment
   - Result: No recommendations field is created
""")

def main():
    """Main function"""
    print("Analyzing GPT response patterns...")
    
    # Analyze GPT responses
    analyze_gpt_responses()
    
    # Analyze extraction logic
    analyze_extraction_logic()

if __name__ == "__main__":
    main()
