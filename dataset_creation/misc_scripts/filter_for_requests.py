from pymongo import MongoClient

# Connect to MongoDB (adjust the database and collection names)
client = MongoClient('localhost', 27017)
db = client['reddit_data']
collection = db['[3]classified_submissions_filtered']

# Filter for entries where either "title" or "selftext" contains "IIL" or "WEWIL"
query = {
    "$or": [
        {"title": {"$regex": "IIL|WEWIL", "$options": "i"}},
        {"selftext": {"$regex": "IIL|WEWIL", "$options": "i"}}
    ]
}

# Fetch filtered data from MongoDB
filtered_data = list(collection.find(query))

# Insert filtered data into a new collection (if needed)
filtered_collection = db['[4]classified_submissions_requests_IIL']
filtered_collection.insert_many(filtered_data)

print("Data filtered and saved to new MongoDB collection successfully.")
