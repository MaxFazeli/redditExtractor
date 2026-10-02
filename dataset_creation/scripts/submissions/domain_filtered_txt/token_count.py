import os
import tiktoken

# Input folder is the current working directory
input_folder = os.getcwd()

# Output folder to store the token count results
output_folder = "token_counts"

# Create the output folder if it doesn't exist
if not os.path.exists(output_folder):
    os.makedirs(output_folder)

total_tokens = 0

# Get an encoding instance for the desired model (adjust the model as needed)
encoding = tiktoken.encoding_for_model("gpt-4")

# Process each .txt file in the input folder
for filename in os.listdir(input_folder):
    if filename.endswith(".txt"):
        file_path = os.path.join(input_folder, filename)
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                content = f.read()
                token_count = len(encoding.encode(content))
                total_tokens += token_count

            # Create an output file for this specific input file
            output_filename = os.path.splitext(filename)[0] + "_token_count.txt"
            output_file_path = os.path.join(output_folder, output_filename)
            with open(output_file_path, "w", encoding="utf-8") as out_f:
                out_f.write(f"File: {filename}\n")
                out_f.write(f"Token count: {token_count}\n")

            print(f"Processed {filename}: {token_count} tokens")
        except Exception as e:
            print(f"Error reading {filename}: {e}")

# Write overall summary file with total token count
summary_file = os.path.join(output_folder, "total_tokens.txt")
with open(summary_file, "w", encoding="utf-8") as sum_f:
    sum_f.write(f"Total tokens across all files: {total_tokens}\n")

print(f"\nToken count files created in '{output_folder}' folder.")
