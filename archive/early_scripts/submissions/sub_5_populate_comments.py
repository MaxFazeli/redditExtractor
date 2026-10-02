from pymongo import MongoClient

# Connect to MongoDB
client = MongoClient('localhost', 27017)
db = client['reddit_data']

# Collections
comments_collection = db['[1]comments_no_null']
submissions_collection = db['[4]submissions_final']

# Fetch all submissions
submissions = list(submissions_collection.find())

# Iterate over all submissions and fetch corresponding comments
for submission in submissions:
    # Extract submission id
    submission_id = submission['id']

    # Find all comments where the parent_id corresponds to this submission_id
    comments = list(comments_collection.find({"parent_id": f"t3_{submission_id}"}))

    # If comments exist, add them to the 'comments' field of the submission
    if comments:
        submission['comments'] = comments

        # Update the submission in the original collection with the 'comments' field
        submissions_collection.update_one(
            {"id": submission_id},  # Match the submission by its id
            {"$set": {"comments": submission['comments']}},
            upsert=True  # If the submission doesn't exist, insert it
        )

print("Comments successfully added to corresponding submissions in the original collection.")
