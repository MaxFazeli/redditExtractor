from pymongo import MongoClient

# Connect to MongoDB
client = MongoClient('mongodb://localhost:27017/')
db = client['reddit_data']
filtered_collection = db['[5]submissions_comments']
top_500_collection = db['[6]top_500_submissions_per_domain']

# Get distinct domains from the collection
domains = filtered_collection.distinct('link_flair_text')

# Iterate over each domain to fetch the top 500 entries
for domain in domains:
    # Fetch the top 500 entries for the current domain sorted by `created_utc` and `score`
    top_submissions = list(
        filtered_collection.find({"link_flair_text": domain})
        .sort([("created_utc", -1), ("score", -1)])
        .limit(500)
    )

    # Insert the top 500 entries into the new collection
    if top_submissions:
        top_500_collection.insert_many(top_submissions)

print("Top 500 submissions for each domain stored successfully.")
