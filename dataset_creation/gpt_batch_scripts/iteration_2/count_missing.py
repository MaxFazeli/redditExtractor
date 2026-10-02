from pymongo import MongoClient
from collections import defaultdict

client     = MongoClient("localhost", 27017)
db         = client["reddit_data"]
collection = db["[12]final_data_it2"]

# Initialize counters
comment_missing_recs = defaultdict(int)
submission_missing_recs = defaultdict(int)
total_comments = defaultdict(int)
total_submissions = defaultdict(int)


def count_missing():
    for doc in collection.find():
        domain = doc.get('link_flair_text', 'Unknown')
        # Count total submissions
        total_submissions[domain] += 1

        # Check submission_recommendations
        if 'submission_recommendations' not in doc:
            submission_missing_recs[domain] += 1

        # Check comments for missing recommendations field
        comments = doc.get('comments', [])
        total_comments[domain] += len(comments)
        for comment in comments:
            if 'recommendations' not in comment:
                comment_missing_recs[domain] += 1

    # Output results
    print("Comments missing 'recommendations' per domain:")
    for domain in sorted(total_comments):
        missing = comment_missing_recs.get(domain, 0)
        total = total_comments[domain]
        print(f"- {domain}: {missing}/{total}")

    print("\nEntries missing 'submission_recommendations' per domain:")
    for domain in sorted(total_submissions):
        missing = submission_missing_recs.get(domain, 0)
        total = total_submissions[domain]
        print(f"- {domain}: {missing}/{total}")


if __name__ == '__main__':
    count_missing()