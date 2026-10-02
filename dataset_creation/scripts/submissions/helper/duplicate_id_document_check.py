import pymongo

# Connect to MongoDB
client = pymongo.MongoClient('localhost', 27017)
db = client['reddit_data']
# Change this if you want to run the check on a different collection
collection = db['[9]final_data_for_labeling']

duplicates_found = False

# Iterate over each document in the collection
for doc in collection.find():
    # Ensure the document has an "id" field for the submission
    if 'id' not in doc:
        continue

    keys_seen = set()
    duplicate_keys = []

    # Generate the submission key
    submission_key = "s_" + doc['id']
    keys_seen.add(submission_key)

    # If the document has a "comments" array, process each comment
    if 'comments' in doc and isinstance(doc['comments'], list):
        for comment in doc['comments']:
            if 'id' not in comment:
                continue
            comment_key = "c_" + comment['id']
            if comment_key in keys_seen:
                duplicate_keys.append(comment_key)
            else:
                keys_seen.add(comment_key)

    if duplicate_keys:
        duplicates_found = True
        print(f"Document with submission id {doc['id']} has duplicate keys: {duplicate_keys}")

if not duplicates_found:
    print("No duplicate keys found in the collection.")
