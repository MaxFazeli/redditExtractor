from openai import OpenAI
client = OpenAI(api_key="")

batch_input_file_id = ""
batch = client.batches.create(
    input_file_id=batch_input_file_id,
    endpoint="/v1/chat/completions",
    completion_window="24h",
    metadata={
        "description": "books classification"
    }
)

print(batch)