import time
from ollama import Client
from tabulate import tabulate

# Function to format the time
def format_time(timestamp):
    return time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(timestamp))

# Function to chat with the model
def chat_with_model(client, model, messages, options={"num_predict": 100}):
    response = client.chat(
        model=model,
        messages=messages,
        options=options
    )
    return response['message']['content']

def main():
    # Create a client instance
    client = Client(host='http://localhost:11434')

    # Pull the specified model from the server
    client.pull('llama3.1')

    # Initial message list
    messages = []

    while True:
        # Get user input
        user_input = input("User: ")

        # Add user message to the list
        messages.append({'role': 'user', 'content': user_input})

        # Start time for response
        start_time = time.time()

        # Get the response from the model
        response_content = chat_with_model(client, 'llama3.1', messages)

        # End time for response
        end_time = time.time()

        # Append model's message to the conversation
        messages.append({'role': 'model', 'content': response_content})

        # Print the chat exchange: user input and bot response
        print(f"Assistant: {response_content}")

        # Print the elapsed time for this exchange in a table format
        elapsed_time = end_time - start_time
        meta_info = [
            ["Elapsed Time (seconds)", f"{elapsed_time:.2f}"]
        ]

        print("\nMeta Info:")
        print(tabulate(meta_info, tablefmt="grid"))

        # Divider for clarity
        print("\n----------------------------------------------------\n")

        # Check if user wants to exit
        if user_input.lower() in ["exit", "quit", "bye"]:
            break

if __name__ == "__main__":
    main()
