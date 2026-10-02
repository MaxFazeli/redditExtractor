import json

input_path = "../../scripts/submissions/domain_filtered_txt/[10]_Music.txt"
system_prompt_path = "../../gpt_prompts/final_prompts/music_prompt_final.txt"
output_file_pattern = "music_batch_{}.jsonl"

with open(system_prompt_path, "r", encoding="utf-8") as f:
    system_prompt = f.read().strip()

with open(input_path, "r", encoding="utf-8") as f:
    data = json.load(f)

if not isinstance(data, list):
    data = [data]

batch_size = 50

for i in range(0, len(data), batch_size):
    current_batch = data[i:i + batch_size]
    batch_number = i // batch_size + 1
    output_path = output_file_pattern.format(batch_number)

    with open(output_path, "w", encoding="utf-8") as outfile:
        for entry in current_batch:

            s_key = next((key for key in entry if key.startswith("s_")), None)
            if s_key is None:
                continue

            custom_id = s_key[2:]
            user_content = json.dumps(entry)

            request = {
                "custom_id": custom_id,
                "method": "POST",
                "url": "/v1/chat/completions",
                "body": {
                    "model": "gpt-4o-mini-2024-07-18",
                    "messages": [
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_content}
                    ],
                    "max_tokens": 16384
                }
            }

            outfile.write(json.dumps(request) + "\n")

    print(f"Written batch {batch_number} with {len(current_batch)} entries to {output_path}")
