import os

# Input folder containing the txt files (assumed to be in the current working directory)
input_folder = os.getcwd()

# Output folder to store the character count results
output_folder = "character_counts"

# Create the output folder if it doesn't exist
if not os.path.exists(output_folder):
    os.makedirs(output_folder)

total_characters = 0

# Process each .txt file in the input folder
for filename in os.listdir(input_folder):
    if filename.endswith(".txt"):
        file_path = os.path.join(input_folder, filename)
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                content = f.read()
                char_count = len(content)
                total_characters += char_count

            # Create an output file for this specific input file
            output_filename = os.path.splitext(filename)[0] + "_char_count.txt"
            output_file_path = os.path.join(output_folder, output_filename)
            with open(output_file_path, "w", encoding="utf-8") as out_f:
                out_f.write(f"File: {filename}\n")
                out_f.write(f"Character count: {char_count}\n")

            print(f"Processed {filename}: {char_count} characters")
        except Exception as e:
            print(f"Error reading {filename}: {e}")

# Write overall summary file with total character count
summary_file = os.path.join(output_folder, "total_characters.txt")
with open(summary_file, "w", encoding="utf-8") as sum_f:
    sum_f.write(f"Total characters across all files: {total_characters}\n")

print(f"\nCharacter count files created in '{output_folder}' folder.")
