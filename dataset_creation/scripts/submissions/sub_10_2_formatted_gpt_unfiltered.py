from pymongo import MongoClient

######################################################
#
# To reduce amount of characters for gpt API, remove
# all fields that are not needed for labeling.
#
######################################################

# Connect to MongoDB
client = MongoClient('localhost', 27017)
db = client['reddit_data']
collection = db['[9_2]final_data_for_labeling_unfiltered']

# Dictionary to hold data for different collections
collections_data = {}

# Fetch and process data
for doc in collection.find():
    if "link_flair_text" not in doc:
        continue  # Skip if link_flair_text is missing

    link_flair = doc["link_flair_text"].replace(" ", "")  # Remove spaces
    collection_name = f"[10]_{link_flair}_unfiltered"

    # Create a flat dictionary with prefixed keys:
    # "s_" for the submission and "c_" for each comment.
    formatted_doc = {}
    # Add submission text with "s_" prefix
    formatted_doc[f"s_{doc['id']}"] = f"{doc.get('title', '')} {doc.get('selftext', '')}".strip()

    # Add each comment with "c_" prefix
    for comment in doc.get("comments", []):
        formatted_doc[f"c_{comment['id']}"] = comment.get("body", "")

    # Store in appropriate collection
    if collection_name not in collections_data:
        collections_data[collection_name] = []
    collections_data[collection_name].append(formatted_doc)

# Insert data into respective collections
for collection_name, docs in collections_data.items():
    db[collection_name].insert_many(docs)

print("Data formatted and stored in respective collections successfully.")
