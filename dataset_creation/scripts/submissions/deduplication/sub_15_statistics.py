from pymongo import MongoClient
from collections import defaultdict, Counter

# Connect to MongoDB
client = MongoClient('mongodb://localhost:27017/')
db = client['reddit_data']
collection = db['[15]submissions_cleaned_full']

# Categories of interest
categories = ['Books', 'Film', 'TV', 'Music', 'Games']

# Initialize counters
stats = {
    'total': 0,
    'gte5_cleaned': 0,
    'gte1_duplicates': 0,
    'total_duplicates_size': 0,
    'category_stats': defaultdict(lambda: {
        'total': 0,
        'gte5_cleaned': 0,
        'gte1_duplicates': 0,
        'total_duplicates_size': 0
    })
}

# Iterate through all submissions
for submission in collection.find():
    stats['total'] += 1
    category = submission.get('link_flair_text')

    cleaned = submission.get('final_recommendations', [])
    duplicates = submission.get('duplicates', [])

    if len(cleaned) >= 5:
        stats['gte5_cleaned'] += 1
    if len(duplicates) >= 1:
        stats['gte1_duplicates'] += 1
    stats['total_duplicates_size'] += len(duplicates)

    if category in categories:
        stats['category_stats'][category]['total'] += 1
        if len(cleaned) >= 5:
            stats['category_stats'][category]['gte5_cleaned'] += 1
        if len(duplicates) >= 1:
            stats['category_stats'][category]['gte1_duplicates'] += 1
        stats['category_stats'][category]['total_duplicates_size'] += len(duplicates)

# Write to file
with open("submission_stats_summary.txt", "w", encoding="utf-8") as file:
    file.write("=== Overall Stats ===\n")
    file.write(f"Total entries: {stats['total']}\n")
    file.write(f"Entries with ≥5 cleaned_recommendations: {stats['gte5_cleaned']}\n")
    file.write(f"Entries with ≥1 duplicates_removed: {stats['gte1_duplicates']}\n")
    avg_dups = stats['total_duplicates_size'] / stats['total'] if stats['total'] else 0
    file.write(f"Average duplicates_removed size: {avg_dups:.2f}\n\n")

    file.write("=== Per Category Stats ===\n")
    for cat in categories:
        cat_stats = stats['category_stats'][cat]
        file.write(f"\nCategory: {cat}\n")
        file.write(f"  Total entries: {cat_stats['total']}\n")
        file.write(f"  Entries with ≥5 cleaned_recommendations: {cat_stats['gte5_cleaned']}\n")
        file.write(f"  Entries with ≥1 duplicates_removed: {cat_stats['gte1_duplicates']}\n")
        avg_cat_dups = (
            cat_stats['total_duplicates_size'] / cat_stats['total']
            if cat_stats['total'] else 0
        )
        file.write(f"  Average duplicates_removed size: {avg_cat_dups:.2f}\n")

print("Statistics written to submission_stats_summary.txt")
