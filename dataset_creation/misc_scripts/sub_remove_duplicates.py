from pymongo import MongoClient

# Connect to MongoDB (adjust the database and collection names)
client = MongoClient('localhost', 27017)
db = client['reddit_data']
collection = db['submissions_filtered']

# Define which fields to check for duplicates
fields_to_check = ['id', 'name', 'url', 'score', 'title', 'created_utc', 'permalink']

# Create a unique set of keys
unique_entries = set()
duplicates_removed = []

# Iterate through the collection
for doc in collection.find():
    # Create a tuple of the fields to check
    key = tuple(doc.get(field, None) for field in fields_to_check)

    # If the combination of fields is unique, keep it
    if key not in unique_entries:
        unique_entries.add(key)
        duplicates_removed.append(doc)

# Optional: Remove the '_id' field before reinserting to avoid duplicate key errors
for doc in duplicates_removed:
    if '_id' in doc:
        del doc['_id']

# Save the deduplicated data back into MongoDB collection
collection_deduped = db['submissions_deduped']
if duplicates_removed:
    collection_deduped.insert_many(duplicates_removed)

print(f"{len(duplicates_removed)} unique entries inserted into 'submissions_deduped' collection.")
