from pymongo import MongoClient

# Connect to MongoDB (adjust the database and collection names)
client = MongoClient('localhost', 27017)
db = client['reddit_data']
collection = db['[1]submissions_no_null']

# Filter out entries where 'score' <= 10, 'over_18' is False, and 'num_comments' <= 5
query = {"$and": [
    {"score": {"$gt": 5}},
    {"over_18": False},
    {"num_comments": {"$gte": 5}},
    {"link_flair_text": {'$ne': None} }],
    "$or": [
        {"title": {"$regex": "IIL|WEWIL", "$options": "i"}},
        {"selftext": {"$regex": "IIL|WEWIL", "$options": "i"}}
    ]
}

# Fetch filtered data from MongoDB
filtered_data = list(collection.find(query))

# Insert filtered data into a new collection (if needed)
filtered_collection = db['[2]submissions_filtered']
filtered_collection.insert_many(filtered_data)

print("Data filtered and saved to new MongoDB collection successfully.")
