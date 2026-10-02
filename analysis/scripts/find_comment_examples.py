#!/usr/bin/env python3
"""
Script to find and display examples of comments with different recommendation states
"""

from pymongo import MongoClient
import json

def find_comment_examples(collection):
    """Find examples of comments with different recommendation states"""
    print("Searching for comment examples...")
    
    examples = {
        'no_field': [],
        'empty_field': [],
        'with_recommendations': []
    }
    
    count = 0
    for doc in collection.find():
        if count >= 10:  # Limit to first 10 submissions for examples
            break
            
        domain = doc.get('link_flair_text', 'Unknown')
        submission_id = doc.get('id', 'Unknown')
        
        comments = doc.get('comments', [])
        for comment in comments:
            comment_id = comment.get('id', 'Unknown')
            comment_body = comment.get('body', '')[:200] + "..." if len(comment.get('body', '')) > 200 else comment.get('body', '')
            
            example = {
                'domain': domain,
                'submission_id': submission_id,
                'comment_id': comment_id,
                'comment_body': comment_body,
                'has_recommendations_field': 'recommendations' in comment,
                'recommendations_value': comment.get('recommendations', 'FIELD_NOT_EXISTS')
            }
            
            if 'recommendations' not in comment:
                if len(examples['no_field']) < 3:  # Get 3 examples
                    examples['no_field'].append(example)
            elif not comment['recommendations'] or len(comment['recommendations']) == 0:
                if len(examples['empty_field']) < 3:  # Get 3 examples
                    examples['empty_field'].append(example)
            else:
                if len(examples['with_recommendations']) < 3:  # Get 3 examples
                    examples['with_recommendations'].append(example)
        
        count += 1
    
    return examples

def display_examples(examples):
    """Display the found examples"""
    print("\n" + "="*80)
    print("COMMENT EXAMPLES FROM DATABASE")
    print("="*80)
    
    # Comments with no recommendations field
    print("\nCOMMENTS WITH NO 'recommendations' FIELD:")
    print("-" * 50)
    for i, example in enumerate(examples['no_field'], 1):
        print(f"\nExample {i}:")
        print(f"  Domain: {example['domain']}")
        print(f"  Submission ID: {example['submission_id']}")
        print(f"  Comment ID: {example['comment_id']}")
        print(f"  Comment Body: {example['comment_body']}")
        print(f"  Has recommendations field: {example['has_recommendations_field']}")
        print(f"  Recommendations value: {example['recommendations_value']}")
    
    # Comments with empty recommendations field
    print("\nCOMMENTS WITH EMPTY 'recommendations' FIELD:")
    print("-" * 50)
    for i, example in enumerate(examples['empty_field'], 1):
        print(f"\nExample {i}:")
        print(f"  Domain: {example['domain']}")
        print(f"  Submission ID: {example['submission_id']}")
        print(f"  Comment ID: {example['comment_id']}")
        print(f"  Comment Body: {example['comment_body']}")
        print(f"  Has recommendations field: {example['has_recommendations_field']}")
        print(f"  Recommendations value: {example['recommendations_value']}")
    
    # Comments with actual recommendations
    print("\nCOMMENTS WITH ACTUAL RECOMMENDATIONS:")
    print("-" * 50)
    for i, example in enumerate(examples['with_recommendations'], 1):
        print(f"\nExample {i}:")
        print(f"  Domain: {example['domain']}")
        print(f"  Submission ID: {example['submission_id']}")
        print(f"  Comment ID: {example['comment_id']}")
        print(f"  Comment Body: {example['comment_body']}")
        print(f"  Has recommendations field: {example['has_recommendations_field']}")
        print(f"  Recommendations value: {example['recommendations_value']}")

def main():
    """Main function to find and display comment examples"""
    print("Finding comment examples from database...")
    
    # Connect to MongoDB
    try:
        client = MongoClient('mongodb://localhost:27017/')
        db = client['reddit_data']
        collection = db['[15]submissions_cleaned_full']
        print("[OK] Connected to MongoDB successfully")
    except Exception as e:
        print(f"[FAIL] Error connecting to MongoDB: {e}")
        return
    
    # Find examples
    examples = find_comment_examples(collection)
    
    # Display examples
    display_examples(examples)
    
    print(f"\nSUMMARY:")
    print(f"  Comments with no field: {len(examples['no_field'])} examples found")
    print(f"  Comments with empty field: {len(examples['empty_field'])} examples found")
    print(f"  Comments with recommendations: {len(examples['with_recommendations'])} examples found")

if __name__ == "__main__":
    main()
