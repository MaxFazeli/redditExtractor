import os
import time
import concurrent.futures
from datetime import datetime
from openai import OpenAI

client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])

def log(message, file_name=None):
    timestamp = datetime.now().strftime("%d-%m-%Y %H:%M:%S")
    if file_name:
        print(f"[{timestamp}] [{file_name}] {message}")
    else:
        print(f"[{timestamp}] {message}")

def process_batch_file(file_path):
    base_name = os.path.basename(file_path)
    log("Processing file...", base_name)

    # Step 1: Upload batch file
    log("[1/4] Uploading batch file...", base_name)
    with open(file_path, "rb") as f:
        batch_input_file = client.files.create(
            file=f,
            purpose="batch"
        )
    log(f"Uploaded file: {batch_input_file}", base_name)
    batch_input_file_id = batch_input_file.id

    # Step 2: Create the batch job
    log("[2/4] Creating batch job...", base_name)
    batch = client.batches.create(
        input_file_id=batch_input_file_id,
        endpoint="/v1/chat/completions",
        completion_window="24h",
        metadata={"description": "classification"}
    )
    log(f"Batch created: {batch}", base_name)
    batch_id = batch.id

    # Step 3: Poll for batch status until completed or failed
    log(f"[3/4] Waiting for batch job ({batch_id}) to complete...", base_name)
    while True:
        batch_status = client.batches.retrieve(batch_id)
        status = batch_status.status
        log(f"Current status: {status}", base_name)
        if status in ["completed", "failed"]:
            break
        time.sleep(20)

    log(f"Final batch status: {batch_status}", base_name)

    # Step 4: Retrieve results (if completed) and write to output file
    if batch_status.status == "completed":
        output_file_id = batch_status.output_file_id
        file_response = client.files.content(output_file_id)
        output_dir = "outputs"
        os.makedirs(output_dir, exist_ok=True)
        output_file_path = os.path.join(output_dir, f"batch_result_{base_name}")
        with open(output_file_path, "w", encoding="utf-8") as outfile:
            outfile.write(file_response.text)
        log(f"[4/4] Result has been written to {output_file_path}", base_name)
    else:
        log(f"Batch job failed with status: {batch_status.status}", base_name)

def main():
    batches_folder = "batches"
    jsonl_files = [os.path.join(batches_folder, f) for f in os.listdir(batches_folder) if f.endswith(".jsonl")]

    max_workers = 10
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(process_batch_file, file): file for file in jsonl_files}
        for future in concurrent.futures.as_completed(futures):
            file = futures[future]
            try:
                future.result()
            except Exception as exc:
                log(f"File {file} generated an exception: {exc}")

if __name__ == "__main__":
    main()
