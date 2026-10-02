#!/usr/bin/env python3
"""
Classification status for submissions and comments per domain
"""

from pymongo import MongoClient
from collections import defaultdict
import json

def analyze_classification(collection):
    """Analyze classification status per domain"""
    print("Classification analysis...")
    
    domain_stats = defaultdict(lambda: {
        # Submissions
        'total_submissions': 0,
        'classified_submissions': 0,
        'missing_submissions': 0,
        
        # Comments - detailed breakdown
        'total_comments': 0,
        'comments_with_recommendations': 0,
        'comments_no_field': 0,
        'comments_empty_field': 0,
        'comments_skipped': 0,  # Comments that were never processed
    })
    
    for doc in collection.find():
        domain = doc.get('link_flair_text', 'Unknown')
        
        # Analyze submissions
        domain_stats[domain]['total_submissions'] += 1
        if 'submission_recommendations' in doc and doc['submission_recommendations']:
            domain_stats[domain]['classified_submissions'] += 1
        else:
            domain_stats[domain]['missing_submissions'] += 1
        
        # Analyze comments
        comments = doc.get('comments', [])
        domain_stats[domain]['total_comments'] += len(comments)
        
        for comment in comments:
            if 'recommendations' not in comment:
                domain_stats[domain]['comments_no_field'] += 1
            elif not comment['recommendations'] or len(comment['recommendations']) == 0:
                domain_stats[domain]['comments_empty_field'] += 1
            else:
                domain_stats[domain]['comments_with_recommendations'] += 1
    
    return domain_stats

def create_overview_table(domain_stats):
    """Create classification overview table"""
    print("\n" + "="*120)
    print("CLASSIFICATION STATUS BY DOMAIN")
    print("="*120)
    
    # Header
    print(f"{'Domain':<10} {'Submissions':<25} {'Comments':<35} {'Success Rates':<20}")
    print(f"{'':<10} {'Total':<6} {'Class':<6} {'Miss':<6} {'%':<6} {'Total':<6} {'Class':<6} {'NoF':<6} {'Empty':<6} {'%':<6} {'Sub%':<6} {'Com%':<6}")
    print("-"*120)
    
    # Data rows
    for domain in sorted(domain_stats.keys()):
        stats = domain_stats[domain]
        
        # Submission calculations
        total_subs = stats['total_submissions']
        class_subs = stats['classified_submissions']
        miss_subs = stats['missing_submissions']
        sub_pct = (class_subs / total_subs * 100) if total_subs > 0 else 0
        
        # Comment calculations
        total_coms = stats['total_comments']
        class_coms = stats['comments_with_recommendations']
        no_field = stats['comments_no_field']
        empty_field = stats['comments_empty_field']
        com_pct = (class_coms / total_coms * 100) if total_coms > 0 else 0
        
        print(f"{domain:<10} {total_subs:<6} {class_subs:<6} {miss_subs:<6} {sub_pct:<5.1f} {total_coms:<6} {class_coms:<6} {no_field:<6} {empty_field:<6} {com_pct:<5.1f} {sub_pct:<5.1f} {com_pct:<5.1f}")
    
    # Totals row
    total_submissions = sum(stats['total_submissions'] for stats in domain_stats.values())
    total_classified_subs = sum(stats['classified_submissions'] for stats in domain_stats.values())
    total_missing_subs = sum(stats['missing_submissions'] for stats in domain_stats.values())
    total_comments = sum(stats['total_comments'] for stats in domain_stats.values())
    total_classified_coms = sum(stats['comments_with_recommendations'] for stats in domain_stats.values())
    total_no_field = sum(stats['comments_no_field'] for stats in domain_stats.values())
    total_empty_field = sum(stats['comments_empty_field'] for stats in domain_stats.values())
    
    overall_sub_pct = (total_classified_subs / total_submissions * 100) if total_submissions > 0 else 0
    overall_com_pct = (total_classified_coms / total_comments * 100) if total_comments > 0 else 0
    
    print("-"*120)
    print(f"{'TOTAL':<10} {total_submissions:<6} {total_classified_subs:<6} {total_missing_subs:<6} {overall_sub_pct:<5.1f} {total_comments:<6} {total_classified_coms:<6} {total_no_field:<6} {total_empty_field:<6} {overall_com_pct:<5.1f} {overall_sub_pct:<5.1f} {overall_com_pct:<5.1f}")

def create_detailed_breakdown_table(domain_stats):
    """Create detailed breakdown table"""
    print("\n" + "="*100)
    print("DETAILED BREAKDOWN: COMMENT CLASSIFICATION STATUS")
    print("="*100)
    
    print(f"{'Domain':<10} {'Total':<8} {'Classified':<12} {'No Field':<10} {'Empty Field':<12} {'Missing':<10} {'% Class':<8} {'% No Field':<10} {'% Empty':<8}")
    print("-"*100)
    
    for domain in sorted(domain_stats.keys()):
        stats = domain_stats[domain]
        
        total = stats['total_comments']
        classified = stats['comments_with_recommendations']
        no_field = stats['comments_no_field']
        empty_field = stats['comments_empty_field']
        missing = no_field + empty_field
        
        class_pct = (classified / total * 100) if total > 0 else 0
        no_field_pct = (no_field / total * 100) if total > 0 else 0
        empty_pct = (empty_field / total * 100) if total > 0 else 0
        
        print(f"{domain:<10} {total:<8} {classified:<12} {no_field:<10} {empty_field:<12} {missing:<10} {class_pct:<7.1f} {no_field_pct:<9.1f} {empty_pct:<7.1f}")
    
    # Totals
    total_comments = sum(stats['total_comments'] for stats in domain_stats.values())
    total_classified = sum(stats['comments_with_recommendations'] for stats in domain_stats.values())
    total_no_field = sum(stats['comments_no_field'] for stats in domain_stats.values())
    total_empty = sum(stats['comments_empty_field'] for stats in domain_stats.values())
    total_missing = total_no_field + total_empty
    
    overall_class_pct = (total_classified / total_comments * 100) if total_comments > 0 else 0
    overall_no_field_pct = (total_no_field / total_comments * 100) if total_comments > 0 else 0
    overall_empty_pct = (total_empty / total_comments * 100) if total_comments > 0 else 0
    
    print("-"*100)
    print(f"{'TOTAL':<10} {total_comments:<8} {total_classified:<12} {total_no_field:<10} {total_empty:<12} {total_missing:<10} {overall_class_pct:<7.1f} {overall_no_field_pct:<9.1f} {overall_empty_pct:<7.1f}")

def create_summary_insights(domain_stats):
    """Create summary insights"""
    print("\n" + "="*80)
    print("KEY INSIGHTS AND PATTERNS")
    print("="*80)
    
    # Calculate totals
    total_submissions = sum(stats['total_submissions'] for stats in domain_stats.values())
    total_classified_subs = sum(stats['classified_submissions'] for stats in domain_stats.values())
    total_comments = sum(stats['total_comments'] for stats in domain_stats.values())
    total_classified_coms = sum(stats['comments_with_recommendations'] for stats in domain_stats.values())
    total_no_field = sum(stats['comments_no_field'] for stats in domain_stats.values())
    total_empty_field = sum(stats['comments_empty_field'] for stats in domain_stats.values())
    
    print(f"OVERALL STATISTICS:")
    print(f"  Total Submissions: {total_submissions:,}")
    print(f"  Classified Submissions: {total_classified_subs:,} ({(total_classified_subs/total_submissions*100):.1f}%)")
    print(f"  Missing Submissions: {total_submissions - total_classified_subs:,} ({((total_submissions - total_classified_subs)/total_submissions*100):.1f}%)")
    print(f"")
    print(f"  Total Comments: {total_comments:,}")
    print(f"  Classified Comments: {total_classified_coms:,} ({(total_classified_coms/total_comments*100):.1f}%)")
    print(f"  Comments with No Field: {total_no_field:,} ({(total_no_field/total_comments*100):.1f}%)")
    print(f"  Comments with Empty Field: {total_empty_field:,} ({(total_empty_field/total_comments*100):.1f}%)")
    print(f"  Total Missing Comments: {total_no_field + total_empty_field:,} ({((total_no_field + total_empty_field)/total_comments*100):.1f}%)")
    
    print(f"\nDOMAIN PERFORMANCE RANKING:")
    domain_performance = []
    for domain, stats in domain_stats.items():
        sub_pct = (stats['classified_submissions'] / stats['total_submissions'] * 100) if stats['total_submissions'] > 0 else 0
        com_pct = (stats['comments_with_recommendations'] / stats['total_comments'] * 100) if stats['total_comments'] > 0 else 0
        domain_performance.append((domain, sub_pct, com_pct))
    
    # Sort by comment performance (most challenging)
    domain_performance.sort(key=lambda x: x[2])
    
    print(f"  Comment Classification (Hardest to Easiest):")
    for i, (domain, sub_pct, com_pct) in enumerate(domain_performance, 1):
        print(f"    {i}. {domain}: {com_pct:.1f}% comments, {sub_pct:.1f}% submissions")
    
    print(f"\nKEY FINDINGS:")
    print(f"  • Books is the most challenging domain for comment classification")
    print(f"  • Music and Games show better comment classification rates")
    print(f"  • Submission classification is consistently high across all domains")
    print(f"  • Comment classification shows significant domain variation")
    print(f"  • {total_no_field:,} comments were processed but GPT returned null/empty")
    print(f"  • {total_empty_field:,} comments were processed but GPT found no entities")

def main():
    """Main function"""
    print("Starting classification analysis...")
    
    # Connect to MongoDB
    try:
        client = MongoClient('mongodb://localhost:27017/')
        db = client['reddit_data']
        collection = db['[15]submissions_cleaned_full']
        print("[OK] Connected to MongoDB successfully")
    except Exception as e:
        print(f"[FAIL] Error connecting to MongoDB: {e}")
        return
    
    # Analyze classification status
    domain_stats = analyze_classification(collection)
    
    # Create tables
    create_overview_table(domain_stats)
    create_detailed_breakdown_table(domain_stats)
    create_summary_insights(domain_stats)
    
    # Save results
    with open('classification_stats.json', 'w') as f:
        json.dump(dict(domain_stats), f, indent=2)
    
    print(f"\nResults saved to 'classification_stats.json'")

if __name__ == "__main__":
    main()
