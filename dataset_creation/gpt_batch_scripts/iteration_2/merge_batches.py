import json
import glob
import os

# Directory containing your JSONL files
input_dir = "result"
pattern = os.path.join(input_dir, "*.jsonl")
output_file = "test/result_2.jsonl"

# Open the merged file for writing
with open(output_file, "w", encoding="utf-8") as out:
    for filepath in glob.glob(pattern):
        with open(filepath, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                # Parse then immediately re-serialize to ensure valid JSONL
                obj = json.loads(line)
                out.write(json.dumps(obj) + "\n")

print(f"Merged lines from {len(glob.glob(pattern))} files into {output_file}")
