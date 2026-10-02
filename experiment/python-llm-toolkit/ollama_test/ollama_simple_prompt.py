from ollama import Client
import time

# Function to format the time
def format_time(timestamp):
    return time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(timestamp))

# Start time print
start_time = time.time()
print(f"Start time: {format_time(start_time)}")

# Create a client instance for connecting to the server at the specified host and port
client = Client(host='http://localhost:11434')

# Pull the specified model from the server
client.pull('llama3.1')

# Send a chat message to the model 'llama3.1' and get a response
response = client.chat(
    model='llama3.1',  # Specify the model name to use for the chat
    messages=[         # List of messages for the chat session
        {
            'role': 'user',   # Role of the message sender (user or system, etc.)
            'content': 'Why is the sky blue?',  # Content of the user's message
        },
    ],
    options={"num_predict": 100}  # Options for the chat request (e.g., number of predictions)
)

# Print the content of the response message from the model
print("\n################### RESPONSE #######################")
print(response['message']['content'])
print("####################################################\n")

# End time print
end_time = time.time()
print(f"End time: {format_time(end_time)}")

# Print the elapsed time
elapsed_time = end_time - start_time
print(f"Elapsed time: {elapsed_time:.2f} seconds")
