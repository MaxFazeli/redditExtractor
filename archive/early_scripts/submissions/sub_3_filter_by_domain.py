from pymongo import MongoClient

# Connect to MongoDB
client = MongoClient('localhost', 27017)
db = client['reddit_data']
collection = db['[2]submissions_filtered']

# Fetch all data
filtered_data = []
for submission in collection.find():
    link_flair_text = submission.get("link_flair_text", "")

    # Remove "- Advanced" if it exists in `link_flair_text`
    if " - Advanced" in link_flair_text:
        base_category = link_flair_text.split(" - Advanced")[0]
        submission["link_flair_text"] = base_category  # Update the field

    filtered_data.append(submission)

# Insert the modified data into a new collection
filtered_collection = db['[3]submissions_unified_domain']
filtered_collection.insert_many(filtered_data)

print("Data processed and saved to new MongoDB collection successfully.")
