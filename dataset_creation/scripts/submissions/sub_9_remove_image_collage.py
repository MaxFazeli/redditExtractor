from pymongo import MongoClient

# Connect to MongoDB
client = MongoClient('localhost', 27017)
db = client['reddit_data']
collection = db['[8]labeling_submissions']

# Excluding submissions where `link_flair_text` is "Image Collage"
filtered_data = []
for submission in collection.find({"link_flair_text": {"$ne": "Image Collage"}}):
    filtered_data.append(submission)

# Insert the filtered data into a new collection
filtered_collection = db['[9]final_data_for_labeling']
filtered_collection.insert_many(filtered_data)

print("Data processed and saved to new MongoDB collection successfully.")
