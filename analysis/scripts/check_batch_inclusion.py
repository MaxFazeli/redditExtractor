#!/usr/bin/env python3
"""
Script to check if comments without recommendations fields were included in batch files
"""

import json
import os
from pymongo import MongoClient

def load_batch_files():
    """Load all batch result files from iterations 1, 2, and 3"""
    batch_data = {
        'iteration_1': {},
        'iteration_2': {},
        'iteration_3': {}
    }
    
    # Iteration 1 batch files
    iteration_1_path = "gpt_batch_scripts/iteration_1/books/outputs"
    if os.path.exists(iteration_1_path):
        for filename in os.listdir(iteration_1_path):
            if filename.endswith('.jsonl'):
                filepath = os.path.join(iteration_1_path, filename)
                try:
                    with open(filepath, 'r', encoding='utf-8') as f:
                        for line in f:
                            if line.strip():
                                data = json.loads(line)
                                custom_id = data.get('custom_id')
                                if custom_id:
                                    batch_data['iteration_1'][custom_id] = data
                except Exception as e:
                    print(f"Error reading {filepath}: {e}")
    
    # Iteration 2 batch files
    iteration_2_path = "gpt_batch_scripts/iteration_2/books/outputs"
    if os.path.exists(iteration_2_path):
        for filename in os.listdir(iteration_2_path):
            if filename.endswith('.jsonl'):
                filepath = os.path.join(iteration_2_path, filename)
                try:
                    with open(filepath, 'r', encoding='utf-8') as f:
                        for line in f:
                            if line.strip():
                                data = json.loads(line)
                                custom_id = data.get('custom_id')
                                if custom_id:
                                    batch_data['iteration_2'][custom_id] = data
                except Exception as e:
                    print(f"Error reading {filepath}: {e}")
    
    # Iteration 3 batch files
    iteration_3_path = "gpt_batch_scripts/iteration_3/books/outputs"
    if os.path.exists(iteration_3_path):
        for filename in os.listdir(iteration_3_path):
            if filename.endswith('.jsonl'):
                filepath = os.path.join(iteration_3_path, filename)
                try:
                    with open(filepath, 'r', encoding='utf-8') as f:
                        for line in f:
                            if line.strip():
                                data = json.loads(line)
                                custom_id = data.get('custom_id')
                                if custom_id:
                                    batch_data['iteration_3'][custom_id] = data
                except Exception as e:
                    print(f"Error reading {filepath}: {e}")
    
    return batch_data

def check_comment_in_batches(comment_id, batch_data):
    """Check if a specific comment ID appears in any batch files"""
    found_in = []
    
    for iteration, batches in batch_data.items():
        for custom_id, batch_result in batches.items():
            # Check if this comment ID appears in the batch result
            if 'response' in batch_result and 'body' in batch_result['response']:
                try:
                    content = batch_result['response']['body']['choices'][0]['message']['content']
                    if comment_id in content:
                        found_in.append(iteration)
                        break
                except:
                    continue
    
    return found_in

def analyze_comment_processing_status(collection, batch_data):
    """Analyze which comments were processed in which iterations"""
    print("Analyzing comment processing status...")
    
    # Get examples of comments without recommendations field
    no_field_examples = []
    empty_field_examples = []
    
    count = 0
    for doc in collection.find():
        if count >= 5:  # Check first 5 submissions
            break
            
        comments = doc.get('comments', [])
        for comment in comments:
            comment_id = comment.get('id', 'Unknown')
            
            if 'recommendations' not in comment:
                if len(no_field_examples) < 3:
                    no_field_examples.append({
                        'comment_id': comment_id,
                        'submission_id': doc.get('id'),
                        'comment_body': comment.get('body', '')[:100] + "..."
                    })
            elif not comment['recommendations'] or len(comment['recommendations']) == 0:
                if len(empty_field_examples) < 3:
                    empty_field_examples.append({
                        'comment_id': comment_id,
                        'submission_id': doc.get('id'),
                        'comment_body': comment.get('body', '')[:100] + "..."
                    })
        
        count += 1
    
    print("\nCOMMENTS WITHOUT RECOMMENDATIONS FIELD:")
    print("-" * 60)
    for i, example in enumerate(no_field_examples, 1):
        print(f"\nExample {i}:")
        print(f"  Comment ID: {example['comment_id']}")
        print(f"  Submission ID: {example['submission_id']}")
        print(f"  Comment Body: {example['comment_body']}")
        
        # Check if this comment was in any batch files
        found_in = check_comment_in_batches(example['comment_id'], batch_data)
        if found_in:
            print(f"  Found in batches: {', '.join(found_in)}")
        else:
            print(f"  NOT FOUND in any batch files")
    
    print("\nCOMMENTS WITH EMPTY RECOMMENDATIONS FIELD:")
    print("-" * 60)
    for i, example in enumerate(empty_field_examples, 1):
        print(f"\nExample {i}:")
        print(f"  Comment ID: {example['comment_id']}")
        print(f"  Submission ID: {example['submission_id']}")
        print(f"  Comment Body: {example['comment_body']}")
        
        # Check if this comment was in any batch files
        found_in = check_comment_in_batches(example['comment_id'], batch_data)
        if found_in:
            print(f"  Found in batches: {', '.join(found_in)}")
        else:
            print(f"  NOT FOUND in any batch files")

def main():
    """Main function to check batch inclusion"""
    print("Checking if comments without recommendations were in batch files...")
    
    # Connect to MongoDB
    try:
        client = MongoClient('mongodb://localhost:27017/')
        db = client['reddit_data']
        collection = db['[15]submissions_cleaned_full']
        print("[OK] Connected to MongoDB successfully")
    except Exception as e:
        print(f"[FAIL] Error connecting to MongoDB: {e}")
        return
    
    # Load batch files
    print("Loading batch files...")
    batch_data = load_batch_files()
    
    print(f"Loaded batch data:")
    for iteration, batches in batch_data.items():
        print(f"  {iteration}: {len(batches)} entries")
    
    # Analyze comment processing status
    analyze_comment_processing_status(collection, batch_data)

if __name__ == "__main__":
    main()
