from openai import OpenAI
client = OpenAI(api_key="")

batch = client.batches.retrieve("batch_67dd3b8011a0819080e8fbe089d50a39")
print(batch)