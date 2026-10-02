from openai import OpenAI
client = OpenAI(api_key="")

file_response = client.files.content("file-FemgRorAvW2hmvaeGebY6D")
print(file_response.text)