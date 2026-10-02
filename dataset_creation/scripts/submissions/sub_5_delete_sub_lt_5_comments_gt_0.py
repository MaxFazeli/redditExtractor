from pymongo import MongoClient

# Connect to MongoDB
client = MongoClient('localhost', 27017)
db = client['reddit_data']

# Collections
submissions_collection = db['[4]submissions_final']
filtered_submissions_collection = db['[5]submissions_comments']

# Fetch submissions with less than 5 comments having positive "score"
submissions_to_store = submissions_collection.find({
    "$expr": {
        "$gt": [
            {
                "$size": {
                    "$filter": {
                        "input": {"$ifNull": ["$comments", []]},
                        "as": "comment",
                        "cond": {"$gt": ["$$comment.score", 0]}
                    }
                }
            },
            4
        ]
    }
})

# Insert the filtered submissions into the new collection
filtered_submissions_collection.insert_many(submissions_to_store)

print("Submissions with at least 5 comments having positive scores were stored in the new collection.")
