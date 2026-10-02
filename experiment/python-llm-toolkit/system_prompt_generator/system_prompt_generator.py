import json
import argparse

def load_system_prompt_config(file_path):
    """
    Load system prompt configuration data from a JSON file.

    :param file_path: str: Path to the JSON file containing input data.
    :return: list: Loaded data as a list of dictionaries.
    """
    with open(file_path, 'r') as file:
        data = json.load(file)
    return data

def generate_system_prompt(role=None, task=None, constraints=None, examples=None, 
                           include_headers=False):
    """
    Generate a system prompt based on role, task, constraints, and optional examples.

    :param role: str, list: Optional role description.
    :param task: str, list: The mandatory task description.
    :param constraints: str, list: The mandatory constraints description.
    :param examples: list of tuple: Optional list of examples in the format (title, body).
    :param include_headers: bool: If True, section headers are included in the prompt.
    :return: str: The generated system prompt.
    :raises ValueError: If task or constraints are not provided.
    """
    # Ensure task and constraints are provided
    if not task or not constraints:
        raise ValueError("Both task and constraints are mandatory.")

    # Helper function to format each section with optional header and handle list or single string
    def format_section(header, content):
        if isinstance(content, list):
            content = "\n".join(content)
        return f"{header}\n{content}\n" if include_headers else f"{content}\n"

    # Build the system prompt
    system_prompt = ""
    
    if role:
        system_prompt += format_section("Role:", role)
    
    system_prompt += format_section("Task:", task)
    system_prompt += format_section("Constraints:", constraints)
    
    if examples:
        examples_text = "Examples:\n" if include_headers else ""
        for example in examples:
            examples_text += '\n'.join(example)
        system_prompt += examples_text

    return system_prompt

def main():
    # Set up the command-line argument parser
    parser = argparse.ArgumentParser(description='Generate system prompts from JSON configuration.')
    
    # Make config_file a required optional argument
    parser.add_argument('--config_file', type=str, required=True, help='Path to the JSON config file containing prompt configurations.')
    
    # The output_file remains optional
    parser.add_argument('--output_file', type=str, help='Path to the output JSON file to save generated prompts.')
    
    # Add an optional argument to control printing to stdout
    parser.add_argument('--print_to_stdout', action='store_true', help='If set, print generated prompts to stdout.')
    
    # Parse command-line arguments
    args = parser.parse_args()

    # Load input data from the specified JSON file
    all_input_data = load_system_prompt_config(args.config_file)

    # List to store all generated prompts with metadata
    generated_prompts = []

    # Iterate over each configuration in the input list
    for input_data in all_input_data:
        # Extract data from the dictionary
        role = input_data.get("role")
        task = input_data["task"]
        constraints = input_data["constraints"]
        examples = input_data.get("examples", [])

        # Generate the system prompt
        system_prompt = generate_system_prompt(role, task, constraints, examples, include_headers=False)
        
        # Gather metadata
        prompt_info = {
            'prompt': system_prompt,
            'contains_role': bool(role),
            'contains_examples': bool(examples)
        }
        
        generated_prompts.append(prompt_info)
        
        # Conditionally print or not print to stdout based on output_file presence and print_to_stdout flag
        if not args.output_file or args.print_to_stdout:
            print(json.dumps(prompt_info, indent=4))  # Print each prompt info as JSON for clarity
            print("="*50)  # Separator for clarity
        
    # Save the generated prompts with metadata to a JSON file if an output path is provided
    if args.output_file:
        with open(args.output_file, 'w') as outfile:
            json.dump(generated_prompts, outfile, indent=4)

if __name__ == "__main__":
    main()