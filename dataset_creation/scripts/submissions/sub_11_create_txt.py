import pymongo
import json
import os

# Define output directory relative to the current working directory
output_dir = os.path.join(os.getcwd(), "domain_filtered_txt")

# Create the directory if it doesn't exist
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# Connect to MongoDB
client = pymongo.MongoClient('localhost', 27017)
db = client['reddit_data']

# Find collections starting with "[10]_"
collection_names = [name for name in db.list_collection_names() if name.startswith("[10]_")]

# Export each collection to its own txt file in the output directory
for coll_name in collection_names:
    # Use a projection to exclude the _id field
    docs = list(db[coll_name].find({}, {'_id': False}))
    file_path = os.path.join(output_dir, f"{coll_name}.txt")
    with open(file_path, "w", encoding="utf-8") as f:
        json.dump(docs, f, indent=4, ensure_ascii=False)

print("Data exported successfully to the 'domain_filtered_txt' folder in the current working directory")
