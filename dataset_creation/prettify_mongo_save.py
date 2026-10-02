import os
import pandas as pd
import json
import io
import concurrent.futures
import pymongo  # Import pymongo for MongoDB connection

encoding = "utf-8"

# MongoDB connection setup
client = pymongo.MongoClient("mongodb://localhost:27017/")
db = client["reddit_data"]  # Replace 'reddit_data' with your database name

# Function to reprocess merged file (if there are merge errors)
def reprocess_merged_file(data_file):
    with io.open(data_file, "r", encoding=encoding) as f:
        content = f.read()

    if "}{" in content:
        fixed_content = content.replace("}{", "}\n{")
        return fixed_content.splitlines()
    else:
        return content.splitlines()

# Function to process data from one file (submissions or comments)
def process_file(data_file):
    print(f"Processing {data_file}")

    # Reprocess the file to handle merge errors
    lines = reprocess_merged_file(data_file)

    data_list = []
    for line in lines:
        try:
            json_data = json.loads(line)
            data_list.append(json_data)
        except json.JSONDecodeError:
            continue

    return data_list

# Function to handle multiple files from a directory
def process_directory(input_dir, collection_name):
    all_data = []
    for root, _, files in os.walk(input_dir):
        for file in files:
            if file.endswith(".txt"):
                file_path = os.path.join(root, file)
                file_data = process_file(file_path)
                all_data.extend(file_data)

    # Convert the data to a DataFrame (optional for further processing)
    df = pd.DataFrame(all_data)

    # Drop columns that contain only NaN values
    df.dropna(axis=1, how='all', inplace=True)

    # Insert data into MongoDB
    if not df.empty:
        records = df.to_dict(orient='records')
        db[collection_name].insert_many(records)
        print(f"Inserted {len(records)} records into {collection_name} collection.")
    else:
        print(f"No data to insert into {collection_name} collection.")

# Function to process both submissions and comments using multithreading
def process_submissions_and_comments(subreddit):
    submission_dir = f"submissions/{subreddit}/"
    comment_dir = f"comments/{subreddit}/"

    submission_collection = f"{subreddit}_submissions"
    comment_collection = f"{subreddit}_comments"

    # Use multithreading to process files in parallel
    with concurrent.futures.ThreadPoolExecutor() as executor:
        futures = []
        futures.append(executor.submit(process_directory, submission_dir, submission_collection))
        futures.append(executor.submit(process_directory, comment_dir, comment_collection))

        for future in concurrent.futures.as_completed(futures):
            future.result()

    print("Processing complete.")

# Main function to trigger the process
def main():
    subreddit = "ifyoulikeblank"
    process_submissions_and_comments(subreddit)

if __name__ == "__main__":
    main()
