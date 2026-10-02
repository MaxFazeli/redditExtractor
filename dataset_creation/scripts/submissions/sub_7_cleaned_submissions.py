from pymongo import MongoClient

# Connect to MongoDB
client = MongoClient('mongodb://localhost:27017/')
db = client['reddit_data']
source_collection = db['[6]top_500_submissions_per_domain']
target_collection = db['[7]cleaned_submissions']

# List of fields to retain
fields_to_keep = {
    "_id", "downs", "link_flair_text", "url", "id", "created_utc", "name", "subreddit",
    "title", "author", "permalink", "selftext", "subreddit_id",
    "ups", "score", "created", "comments"
}

# Fetch all documents from the source collection
documents = source_collection.find()

# Retain only specified fields and store in a new collection
cleaned_documents = []
for doc in documents:
    cleaned_doc = {key: doc[key] for key in fields_to_keep if key in doc}
    cleaned_documents.append(cleaned_doc)

# Insert cleaned documents into the target collection
if cleaned_documents:
    target_collection.insert_many(cleaned_documents)

print("Documents cleaned and stored in a new collection successfully.")
