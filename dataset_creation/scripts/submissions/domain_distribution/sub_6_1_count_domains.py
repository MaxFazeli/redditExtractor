from pymongo import MongoClient
from collections import Counter

# Connect to MongoDB
client = MongoClient('mongodb://localhost:27017/')
db = client['reddit_data']
filtered_collection = db['[6]top_500_submissions_per_domain']

# Fetch all filtered submissions
submissions = filtered_collection.find()

# Extract domain_area values and count occurrences
domain_area_counter = Counter(submission.get('link_flair_text') for submission in submissions)

# Write the distribution to a text file
with open("domains_distribution_6_1.txt", "w") as file:
    file.write("Distribution of domains:\n")
    for domain, count in domain_area_counter.items():
        file.write(f"{domain}: {count}\n")

print("Distribution written to domain_distribution_6_1.txt")
