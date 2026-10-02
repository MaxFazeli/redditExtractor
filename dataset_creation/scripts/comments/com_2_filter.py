from pymongo import MongoClient

# Connect to MongoDB (adjust the database and collection names)
client = MongoClient('localhost', 27017)
db = client['reddit_data']
collection = db['comments_no_null']

# Filter out entries where 'score' <= 0, 'over_18' is True, or 'num_comments' < 3
query = {"score": {"$gt": 0}}

# Fetch filtered data from MongoDB
filtered_data = list(collection.find(query))

# Insert filtered data into a new collection (if needed)
filtered_collection = db['comments_filtered']
filtered_collection.insert_many(filtered_data)

print("Data filtered and saved to new MongoDB collection successfully.")
