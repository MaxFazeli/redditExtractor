#!/usr/bin/env python3
"""
Classification analysis that distinguishes between:
- Comments with no recommendations field
- Comments with empty recommendations
- Comments with actual recommendations
"""

from pymongo import MongoClient
from collections import defaultdict
import json

def analyze_missing_types(collection):
    """Analyze classification status with more granular detail"""
    print("\nAnalysing missing classification types...")
    
    domain_stats = defaultdict(lambda: {
        'total_submissions': 0,
        'classified_submissions': 0,
        'total_comments': 0,
        'comments_no_field': 0,      # No 'recommendations' field
        'comments_empty_field': 0,   # 'recommendations' field exists but is empty/null
        'comments_with_recommendations': 0  # 'recommendations' field has content
    })
    
    for doc in collection.find():
        domain = doc.get('link_flair_text', 'Unknown')
        domain_stats[domain]['total_submissions'] += 1
        
        # Check submission classification
        if 'submission_recommendations' in doc and doc['submission_recommendations']:
            domain_stats[domain]['classified_submissions'] += 1
        
        # Analyze comments in detail
        comments = doc.get('comments', [])
        domain_stats[domain]['total_comments'] += len(comments)
        
        for comment in comments:
            if 'recommendations' not in comment:
                # No recommendations field at all
                domain_stats[domain]['comments_no_field'] += 1
            elif not comment['recommendations'] or len(comment['recommendations']) == 0:
                # Recommendations field exists but is empty/null
                domain_stats[domain]['comments_empty_field'] += 1
            else:
                # Recommendations field has actual content
                domain_stats[domain]['comments_with_recommendations'] += 1
    
    print("\nDetailed classification status:")
    for domain in sorted(domain_stats.keys()):
        stats = domain_stats[domain]
        sub_pct = (stats['classified_submissions'] / stats['total_submissions'] * 100) if stats['total_submissions'] > 0 else 0
        com_pct = (stats['comments_with_recommendations'] / stats['total_comments'] * 100) if stats['total_comments'] > 0 else 0
        
        print(f"\n  {domain}:")
        print(f"    Submissions: {stats['classified_submissions']}/{stats['total_submissions']} ({sub_pct:.1f}%)")
        print(f"    Comments with recommendations: {stats['comments_with_recommendations']}/{stats['total_comments']} ({com_pct:.1f}%)")
        print(f"    Comments with no field: {stats['comments_no_field']}")
        print(f"    Comments with empty field: {stats['comments_empty_field']}")
        print(f"    Total missing: {stats['comments_no_field'] + stats['comments_empty_field']}")
    
    return domain_stats

def main():
    """Main function"""
    print("Starting analysis of missing classification types...")
    
    # Connect to MongoDB
    try:
        client = MongoClient('mongodb://localhost:27017/')
        db = client['reddit_data']
        collection = db['[15]submissions_cleaned_full']
        print("[OK] Connected to MongoDB successfully")
    except Exception as e:
        print(f"[FAIL] Error connecting to MongoDB: {e}")
        return
    
    # Analyze
    domain_stats = analyze_missing_types(collection)
    
    # Calculate overall totals
    total_submissions = sum(stats['total_submissions'] for stats in domain_stats.values())
    classified_submissions = sum(stats['classified_submissions'] for stats in domain_stats.values())
    total_comments = sum(stats['total_comments'] for stats in domain_stats.values())
    comments_with_recommendations = sum(stats['comments_with_recommendations'] for stats in domain_stats.values())
    comments_no_field = sum(stats['comments_no_field'] for stats in domain_stats.values())
    comments_empty_field = sum(stats['comments_empty_field'] for stats in domain_stats.values())
    
    print(f"\nOVERALL TOTALS:")
    print(f"  Total Submissions: {total_submissions}")
    print(f"  Classified Submissions: {classified_submissions} ({classified_submissions/total_submissions*100:.1f}%)")
    print(f"  Total Comments: {total_comments}")
    print(f"  Comments with recommendations: {comments_with_recommendations} ({comments_with_recommendations/total_comments*100:.1f}%)")
    print(f"  Comments with no field: {comments_no_field}")
    print(f"  Comments with empty field: {comments_empty_field}")
    print(f"  Total missing comments: {comments_no_field + comments_empty_field}")

if __name__ == "__main__":
    main()
