FROM python:3.13-bullseye

# Set the working directory
WORKDIR /app

# Print Python output immediately (docker logs) instead of buffering it
ENV PYTHONUNBUFFERED=1
# Same time zone in all logs (run.log, loguru, result timestamps)
ENV TZ=Europe/Vienna

# Experiment dependencies (openssh-client for the cluster connection is part of the base image)
RUN pip install --no-cache-dir \
    pymongo==4.10.1 \
    tqdm==4.67.0 \
    pandas==2.2.2 \
    loguru==0.7.2 \
    ollama==0.3.1 \
    numpy==2.2.0 \
    openai==1.70.0

# The experiment directory is mounted as a volume at runtime (see docker-compose files),
# so code changes do not require rebuilding the image.

# Command to keep the container running (overridden by the compose files)
CMD ["tail", "-f", "/dev/null"]
