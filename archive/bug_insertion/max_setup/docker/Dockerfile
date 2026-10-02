FROM python:3.13-bullseye

# Set the working directory
WORKDIR /app

# Copy the application code
COPY ./cs-master/01_data_structures_and_algorithms_in_python /app/src
COPY ./python-llm-toolkit/sbatch_scheduler /app/sbatch_scheduler
COPY ./scripts /app/scripts
COPY ./prompts /app/prompts

RUN cd src && \
    pip install --no-cache-dir -r requirements.txt

# Command to keep the container running
CMD ["tail", "-f", "/dev/null"]

