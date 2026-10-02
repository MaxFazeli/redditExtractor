from openai import OpenAI
client = OpenAI(api_key="")

batch_input_file = client.files.create(
    file=open("test_books.jsonl", "rb"),
    purpose="batch"
)

print(batch_input_file)