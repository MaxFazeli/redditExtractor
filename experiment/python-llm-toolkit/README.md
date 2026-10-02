# python-llm-toolkit

> Vendored copy of https://github.com/marweb1996/python-llm-toolkit (branch `develop`, commit `e7310af`).
> `sbatch_scheduler/` was modified for the recommendation experiment, see `docs/changelog.md` (section B).

Welcome to the `python-llm-toolkit` repository! This toolkit provides scripts and instructions to efficiently set up and run an Ollama Server on the TU Graz NVCluster, along with guidance on how to remotely connect and execute specific Python scripts. **Please note**: You need to be connected to the TU Graz VPN for access to the NVCluster.

## Introduction

The `python-llm-toolkit` is designed to facilitate researchers and developers in deploying and managing large language model (LLM) tasks using Ollama Server on the TU Graz NVCluster. This toolkit simplifies the process of setting up the server, running Python scripts, and ensuring efficient remote connectivity.

## Prerequisites

Before you begin, ensure you have met the following requirements:

- **Connection to the TU Graz VPN.**
- SSH access to the TU Graz NVCluster.
- Python 3.x installed on your local machine.
- Basic knowledge of using SSH and Python scripting.

## Installation

1. **Clone this repository to your local machine:**

   ```bash
   git clone git@github.com:marweb1996/python-llm-toolkit.git
   cd python-llm-toolkit
   ```
2. **Install Ollama on NVCluster in your home directory (/home/username):**
   ```bash
   wget https://ollama.com/download/ollama-linux-amd64.tgz
   tar -xzf ollama-linux-amd64.tgz
   ```
3. **Run SBATCH Job Scheduler:** 
   ```bash
   cd sbatch_scheduler
   python3 sbatch_scheduler.py --sbatch-script ollama_serve.sh -u <username> -host <username>@nvcluster.tugraz.at --keep-alive --ollama-port=<port-number> --port-forwarding
   ```

   > **Important:** It is crucial to choose a unique port number for the Ollama server to avoid conflicts with ports used by others on the remote host. An effective strategy is to use a number derived from something personal, such as your birthdate, to ensure uniqueness. For example, if your birthdate is 17.03.1995, you can use 1703 and add a trailing zero to make 17030, which falls within the allowable range of 0 to 65535. Avoid using system ports (0-1023), also known as "well-known ports," as these are typically reserved for system processes.

4. **Run Ollama example script (new terminal session):** 
   ```bash
   cd ollama_test
   python3 ollama_simple_prompt.py
   ```