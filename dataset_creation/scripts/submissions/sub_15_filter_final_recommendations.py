from pymongo import MongoClient

# Connect to MongoDB
client = MongoClient('mongodb://localhost:27017/')
db = client['reddit_data']
source_collection = db['[15]submissions_cleaned_full']
target_collection = db['dataset']

# Filter submissions where final_recommendations array has 5 or more entries
filtered_documents = []
kicked_count = 0
for submission in source_collection.find():
    if 'final_recommendations' in submission:
        recommendations = submission.get('final_recommendations', [])
        if isinstance(recommendations, list) and len(recommendations) >= 5:
            filtered_documents.append(submission)
        else:
            kicked_count += 1
    else:
        kicked_count += 1

# Insert filtered documents into the target collection
if filtered_documents:
    target_collection.insert_many(filtered_documents)
    print(f"Successfully inserted {len(filtered_documents)} submissions into 'dataset' collection.")
    print(f"Kicked {kicked_count} submissions due to having less than 5 recommendations.")
else:
    print("No submissions found with 5 or more recommendations.")
    print(f"Kicked {kicked_count} submissions due to having less than 5 recommendations.")

print("Data processed and saved to MongoDB collection successfully.")
