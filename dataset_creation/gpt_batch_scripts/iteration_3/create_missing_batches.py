import json
from pymongo import MongoClient

# Configuration
INPUT_PATH = "../../scripts/submissions/domain_filtered_txt/[10]_Music.txt"
SYSTEM_PROMPT_PATH = "../../gpt_prompts/final_prompts/music_prompt_final.txt"
OUTPUT_FILE_PATTERN = "music/batches/music_batch_missing_{}.jsonl"

# MongoDB setup
client     = MongoClient("localhost", 27017)
db         = client["reddit_data"]
collection = db["[12]final_data_it2"]

# Step 1: Automatically collect IDs of comments missing 'recommendations'
missing_comment_ids = set()
for doc in collection.find():
    for comment in doc.get('comments', []):
        if 'recommendations' not in comment:
            # 'id' field holds the comment ID without prefix
            missing_comment_ids.add(comment['id'])

print(f"Found {len(missing_comment_ids)} comments missing recommendations")

# Load system prompt
with open(SYSTEM_PROMPT_PATH, 'r', encoding='utf-8') as f:
    system_prompt = f.read().strip()

# Load submissions data
with open(INPUT_PATH, 'r', encoding='utf-8') as f:
    data = json.load(f)

if not isinstance(data, list):
    data = [data]

# Step 2: Filter entries to those containing any missing-comment IDs
filtered_entries = []
for entry in data:
    # Find the submission key
    s_key = next((k for k in entry if k.startswith('s_')), None)
    if not s_key:
        continue

    # Identify comment keys whose ID is in missing_comment_ids
    matching_c_keys = [
        k for k in entry
        if k.startswith('c_') and k[2:] in missing_comment_ids
    ]

    if not matching_c_keys:
        continue

    # Build a trimmed entry: only the submission and the missing comments
    trimmed = {s_key: entry[s_key]}
    for c_key in matching_c_keys:
        trimmed[c_key] = entry[c_key]
    filtered_entries.append(trimmed)

print(f"Preparing {len(filtered_entries)} submissions for batching")

# Step 3: Batch and write out
batch_size = 50
for i in range(0, len(filtered_entries), batch_size):
    batch = filtered_entries[i:i + batch_size]
    batch_num = (i // batch_size) + 1
    output_path = OUTPUT_FILE_PATTERN.format(batch_num)

    with open(output_path, 'w', encoding='utf-8') as outfile:
        for entry in batch:
            s_key = next(k for k in entry if k.startswith('s_'))
            custom_id = s_key[2:]

            request = {
                "custom_id": custom_id,
                "method": "POST",
                "url": "/v1/chat/completions",
                "body": {
                    "model": "gpt-4o-mini-2024-07-18",
                    "messages": [
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": json.dumps(entry)}
                    ],
                    "max_tokens": 16384
                }
            }
            outfile.write(json.dumps(request) + "\n")

    print(f"Written batch {batch_num} with {len(batch)} entries to {output_path}")
