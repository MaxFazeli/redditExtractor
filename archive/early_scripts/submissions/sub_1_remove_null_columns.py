from pymongo import MongoClient
import pandas as pd
import math
from concurrent.futures import ThreadPoolExecutor

# Connect to MongoDB
client = MongoClient('localhost', 27017)
db = client['reddit_data']
collection = db['ifyoulikeblank_submissions']

# Function to remove keys with NaN, None, or empty string values
def clean_document(document):
    return {k: v for k, v in document.items() if v not in [None, '', float('nan')] and not (isinstance(v, float) and math.isnan(v))}

# Function to process a batch of documents
def process_batch(batch):
    cleaned_batch = [clean_document(doc) for doc in batch if clean_document(doc)]
    return cleaned_batch

# Fetch documents in batches from MongoDB
batch_size = 10000  # Adjust this for optimal performance based on your environment
cursor = collection.find({})

cleaned_documents = []
batch = []

# Parallel processing using ThreadPoolExecutor
with ThreadPoolExecutor() as executor:
    futures = []
    for doc in cursor:
        batch.append(doc)
        if len(batch) == batch_size:
            futures.append(executor.submit(process_batch, batch))
            batch = []  # Reset batch

    # If there's any remaining batch
    if batch:
        futures.append(executor.submit(process_batch, batch))

    # Gather all cleaned documents
    for future in futures:
        cleaned_documents.extend(future.result())

# Bulk insert cleaned documents into a new collection
collection_cleaned = db['[1]submissions_no_null']
if cleaned_documents:
    collection_cleaned.insert_many(cleaned_documents)

print(f"Data cleaned and inserted into 'submissions_cleaned' collection.")
