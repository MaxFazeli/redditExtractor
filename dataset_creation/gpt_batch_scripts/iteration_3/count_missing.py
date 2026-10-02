from pymongo import MongoClient
from collections import defaultdict

client     = MongoClient("localhost", 27017)
db         = client["reddit_data"]
collection = db["[13]all_fields"]

# Initialize counters
comment_missing_recs = defaultdict(int)
submission_missing_recs = defaultdict(int)
total_comments = defaultdict(int)
total_submissions = defaultdict(int)


def count_missing():
    # Create a dictionary to store missing entries per domain
    missing_entries = defaultdict(lambda: {'comments': [], 'submissions': []})

    for doc in collection.find():
        domain = doc.get('link_flair_text', 'Unknown')
        # Count total submissions
        total_submissions[domain] += 1

        # Check submission_recommendations
        if 'submission_recommendations' not in doc:
            submission_missing_recs[domain] += 1
            missing_entries[domain]['submissions'].append(doc.get('_id'))

        # Check comments for missing recommendations field
        comments = doc.get('comments', [])
        total_comments[domain] += len(comments)
        for comment in comments:
            if 'recommendations' not in comment:
                comment_missing_recs[domain] += 1
                missing_entries[domain]['comments'].append(comment.get('_id'))

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

    # Write missing entries to files per domain
    for domain, entries in missing_entries.items():
        filename = f"{domain}.txt"  # e.g., books.txt, film.txt
        with open(filename, 'w') as file:
            if entries['submissions']:
                file.write("Missing submission recommendations (submission IDs):\n")
                for submission_id in entries['submissions']:
                    file.write(f"{submission_id}\n")

            if entries['comments']:
                file.write("\nMissing recommendations in comments (comment IDs):\n")
                for comment_id in entries['comments']:
                    file.write(f"{comment_id}\n")


if __name__ == '__main__':
    count_missing()
