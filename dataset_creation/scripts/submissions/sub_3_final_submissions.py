from pymongo import MongoClient

# Connect to MongoDB
client = MongoClient('localhost', 27017)
db = client['reddit_data']
collection = db['[3]submissions_unified_domain']
output_collection = db['[4]submissions_final']

# Aggregation pipeline to count entries by `link_flair_text` and filter for counts > 100
pipeline = [
    {"$group": {"_id": "$link_flair_text", "count": {"$sum": 1}}},
    {"$match": {"count": {"$gt": 100}}}
]

# Run the aggregation to find domains with more than 100 entries
frequent_domains = list(collection.aggregate(pipeline))

# Extract the link_flair_text values with more than 100 entries
frequent_domain_values = [domain['_id'] for domain in frequent_domains]

# Query to get all entries that belong to the frequent domains
query = {
    "link_flair_text": {"$in": frequent_domain_values}
}

# Fetch the entries that match the query
entries_to_store = list(collection.find(query))

# Insert the matching entries into the target collection
if entries_to_store:
    output_collection.insert_many(entries_to_store)
    print("Matching entries have been processed and stored in the new MongoDB collection successfully.")
else:
    print("No entries with the specified domains were found.")
