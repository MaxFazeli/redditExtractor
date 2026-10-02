import os
import glob
import json

# List of categories to process
categories = ["books", "games", "movies", "tv", "music"]
# Directory to write the merged result
result_dir = "result"
output_file = os.path.join(result_dir, "result_3.jsonl")

# Ensure result directory exists
os.makedirs(result_dir, exist_ok=True)

with open(output_file, "w", encoding="utf-8") as out:
    total_files = 0
    total_lines = 0

    for category in categories:
        input_dir = os.path.join(category, "outputs")
        pattern = os.path.join(input_dir, "*.jsonl")
        matched_files = glob.glob(pattern)
        total_files += len(matched_files)

        for filepath in matched_files:
            with open(filepath, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    # Validate by parsing and re-dumping
                    try:
                        obj = json.loads(line)
                        out.write(json.dumps(obj) + "\n")
                        total_lines += 1
                    except json.JSONDecodeError:
                        print(f"Skipping invalid JSON in {filepath}: {line[:60]}...")

    print(f"Merged {total_lines} lines from {total_files} files into {output_file}")
