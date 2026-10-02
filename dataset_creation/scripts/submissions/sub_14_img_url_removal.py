from pymongo import MongoClient
import json

# Connect to MongoDB
client = MongoClient('localhost', 27017)
db = client['reddit_data']
collection = db['[13]all_fields']

# Prepare output collection
filtered_collection = db['[14]image_url_removal']

# Open log file for excluded documents
with open('excluded_entries.log', 'w', encoding='utf-8') as log_file:
    filtered_data = []
    excluded_count = 0

    for submission in collection.find():
        submission_recommendations = submission.get('submission_recommendations', None)
        url = submission.get('url', "")

        # Exclusion conditions
        is_null_recommendations = submission_recommendations is None
        is_image_url = False
        if url:
            is_image_url = (".jpg" in url.lower()) or (".png" in url.lower())

        if is_null_recommendations and is_image_url:
            # Log only selected fields
            log_entry = {
                "subreddit_id": submission.get("subreddit_id"),
                "link_flair_text": submission.get("link_flair_text"),
                "title": submission.get("title"),
                "url": submission.get("url"),
                "permalink": submission.get("permalink")
            }
            log_file.write(json.dumps(log_entry, default=str) + '\n')
            excluded_count += 1
        else:
            filtered_data.append(submission)

    # Insert filtered data
    if filtered_data:
        filtered_collection.insert_many(filtered_data)

print(f"Data processed. {excluded_count} entries were excluded and logged.")
