from pymongo import MongoClient

# Connect to MongoDB
client = MongoClient('mongodb://localhost:27017/')
db = client['reddit_data']
source_collection = db['[7_2]cleaned_submissions_unfiltered']
target_collection = db['[8_2]labeling_submissions_unfiltered']

# List of fields to retain
fields_to_keep = {
    "_id","id", "link_flair_text",
    "title","selftext", "score", "comments"
}

fields_to_keep_comments = {
    "_id", "id", "score", "body"
}

# Fetch all documents from the source collection
documents = source_collection.find()

# Retain only specified fields and store in a new collection for reduced API tokens
labeling_documents = []
for doc in documents:
    # Retain only specified fields in the main document
    labeling_doc = {key: doc[key] for key in fields_to_keep if key in doc}

    # Process comments to retain specified fields
    comments = doc.get("comments", [])
    labeling_comments = [
        {key: comment[key] for key in fields_to_keep_comments if key in comment}
        for comment in comments
    ]

    # Add the cleaned comments back to the document
    labeling_doc["comments"] = labeling_comments

    labeling_documents.append(labeling_doc)

# Insert the cleaned documents into the target collection
if labeling_documents:
    target_collection.insert_many(labeling_documents)

print("Documents prepared for labeling and stored in a new collection successfully.")
