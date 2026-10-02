from ollama import Client
from ollama._types import ResponseError
import time
import json

# Function to format the time
def format_time(timestamp):
    return time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(timestamp))

# Start time print
start_time = time.time()
print(f"Start time: {format_time(start_time)}")

# Create a client instance for connecting to the server at the specified host and port
client = Client(host='http://localhost:11434')

# Pull the specified model from the server
model = "gemma:7b-instruct-v1.1-q4_0"
try:
    response = client.show(model)
except ResponseError as e:
    print(f"Model {model} not loaded yet! Pulling...")
    client.pull(model)
    print(f"Finished pulling model {model}.")
    response = client.show(model)

print("\n################### RESPONSE #######################")
print(json.dumps(response, indent=4))
print("####################################################\n")

# End time print
end_time = time.time()
print(f"End time: {format_time(end_time)}")

# Print the elapsed time
elapsed_time = end_time - start_time
print(f"Elapsed time: {elapsed_time:.2f} seconds")
