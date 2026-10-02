import pymongo

# Connect to MongoDB
client = pymongo.MongoClient('localhost', 27017)
db = client['reddit_data']
collection = db['[9]final_data_for_labeling']

submission_keys = set()
duplicates_found = False

# Iterate over each document in the collection
for doc in collection.find():
    # Ensure the document has an "id" field for the submission
    if 'id' not in doc:
        print("Document missing 'id' field:", doc)
        continue

    submission_key = "s_" + doc['id']

    # Check if this submission key is already seen
    if submission_key in submission_keys:
        duplicates_found = True
        print(f"Duplicate submission key found: {submission_key}")
    else:
        submission_keys.add(submission_key)

if not duplicates_found:
    print("All submission keys are unique in the dataset.")
