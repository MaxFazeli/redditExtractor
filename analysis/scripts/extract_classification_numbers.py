#!/usr/bin/env python3
"""
Script to extract exact classification numbers from MongoDB
for each iteration and domain.
"""

from pymongo import MongoClient
from collections import defaultdict
import json

def connect_to_mongodb():
    """Connect to MongoDB and return the collection"""
    try:
        client = MongoClient('mongodb://localhost:27017/')
        db = client['reddit_data']
        collection = db['[15]submissions_cleaned_full']
        print("[OK] Connected to MongoDB successfully")
        return collection
    except Exception as e:
        print(f"[FAIL] Error connecting to MongoDB: {e}")
        return None

def get_total_counts(collection):
    """Get total submissions and comments by domain"""
    print("\nGetting total counts by domain...")
    
    domain_totals = defaultdict(lambda: {'submissions': 0, 'comments': 0})
    
    for doc in collection.find():
        domain = doc.get('link_flair_text', 'Unknown')
        domain_totals[domain]['submissions'] += 1
        
        comments = doc.get('comments', [])
        domain_totals[domain]['comments'] += len(comments)
    
    print("Total counts by domain:")
    for domain in sorted(domain_totals.keys()):
        subs = domain_totals[domain]['submissions']
        comments = domain_totals[domain]['comments']
        print(f"  {domain}: {subs} submissions, {comments} comments")
    
    return domain_totals

def analyze_classification_status(collection):
    """Analyze current classification status"""
    print("\nAnalyzing current classification status...")
    
    domain_stats = defaultdict(lambda: {
        'total_submissions': 0,
        'classified_submissions': 0,
        'total_comments': 0,
        'classified_comments': 0
    })
    
    for doc in collection.find():
        domain = doc.get('link_flair_text', 'Unknown')
        domain_stats[domain]['total_submissions'] += 1
        
        # Check if submission has recommendations
        if 'submission_recommendations' in doc and doc['submission_recommendations']:
            domain_stats[domain]['classified_submissions'] += 1
        
        # Count comments and their classification status
        comments = doc.get('comments', [])
        domain_stats[domain]['total_comments'] += len(comments)
        
        for comment in comments:
            if 'recommendations' in comment and comment['recommendations']:
                domain_stats[domain]['classified_comments'] += 1
    
    print("\nCurrent classification status:")
    for domain in sorted(domain_stats.keys()):
        stats = domain_stats[domain]
        sub_pct = (stats['classified_submissions'] / stats['total_submissions'] * 100) if stats['total_submissions'] > 0 else 0
        com_pct = (stats['classified_comments'] / stats['total_comments'] * 100) if stats['total_comments'] > 0 else 0
        
        print(f"  {domain}:")
        print(f"    Submissions: {stats['classified_submissions']}/{stats['total_submissions']} ({sub_pct:.1f}%)")
        print(f"    Comments: {stats['classified_comments']}/{stats['total_comments']} ({com_pct:.1f}%)")
    
    return domain_stats

def main():
    """Main function to extract classification numbers"""
    print("Starting classification number extraction...")
    
    # Connect to MongoDB
    collection = connect_to_mongodb()
    if collection is None:
        return
    
    # Get total counts
    domain_totals = get_total_counts(collection)
    
    # Analyze current status
    domain_stats = analyze_classification_status(collection)
    
    # Calculate totals
    total_submissions = sum(stats['total_submissions'] for stats in domain_stats.values())
    total_classified_submissions = sum(stats['classified_submissions'] for stats in domain_stats.values())
    total_comments = sum(stats['total_comments'] for stats in domain_stats.values())
    total_classified_comments = sum(stats['classified_comments'] for stats in domain_stats.values())
    
    print(f"\nOVERALL TOTALS:")
    print(f"  Total Submissions: {total_submissions}")
    print(f"  Classified Submissions: {total_classified_submissions} ({total_classified_submissions/total_submissions*100:.1f}%)")
    print(f"  Total Comments: {total_comments}")
    print(f"  Classified Comments: {total_classified_comments} ({total_classified_comments/total_comments*100:.1f}%)")
    
    # Save results to file
    results = {
        'domain_totals': dict(domain_totals),
        'domain_stats': dict(domain_stats),
        'overall_totals': {
            'total_submissions': total_submissions,
            'classified_submissions': total_classified_submissions,
            'total_comments': total_comments,
            'classified_comments': total_classified_comments
        }
    }
    
    with open('mongodb_classification_results.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\nResults saved to 'mongodb_classification_results.json'")

if __name__ == "__main__":
    main()
