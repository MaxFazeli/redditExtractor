# SBATCH Job Scheduler

This project provides a Python script to submit and monitor SBATCH jobs on a remote host. The script allows you to:
- Submit an SBATCH job to a remote cluster.
- Monitor the job's status until it starts running.
- Forward a local port to the service started by the job (SSH tunnel), and wait until it answers.
- Run a local script, and stop the SBATCH job afterwards (also on Ctrl+C / `docker stop`).

> Modified for the recommendation experiment (see `docs/changelog.md` in the
> thesis repository): robust job monitoring, tunnel readiness check and auto-restart, SIGTERM handling,
> per-port remote script names, exit-code propagation.

## Prerequisites

1. **Python Environment**: Ensure you have Python 3.x installed.
2. **SSH Access**: Set up SSH keys for passwordless access to the remote cluster (ssh-copy-id). All ssh calls run with `BatchMode=yes`, so a missing key fails immediately instead of waiting for a password.

## Usage

### Example Command
```sh
python3 sbatch_scheduler.py --sbatch-script ollama_serve.sh -u <username> -host <username>@nvcluster.tugraz.at --keep-alive --ollama-port=<port-number> --port-forwarding
```

### Arguments

- `--sbatch-script`: The local SBATCH script template. It is uploaded as `<remote-dir>/<name>_<ollama-port>.sh`, so several instances with different ports do not overwrite each other.
- `--remote-sbatch-script`: The path to an SBATCH script already on the remote host.
- `--sbatch-script-dir`: Directory on remote host where SBATCH script should be created (and the job submitted from, i.e. where the logs end up). Defaults to the user's home directory.
- `--skip-sbatch-script-creation`: Skip creating the SBATCH script on the remote host if it already exists.
- `--script` (optional): The `.py` or `.sh` script to run locally once the job runs (and, with port forwarding, Ollama answers). Its exit code becomes the exit code of the scheduler.
- `-u, --user`: The user name for filtering jobs in the queue (required).
- `-host, --host`: The remote host in the format `user@host[:port]` (required).
- `--ollama-port`: Port written into the SBATCH script (`PORT_NUMBER=`). Must be 1024–65535 and unique among running instances. Default `11434`.
- `--local-ollama-port`: Local end of the tunnel. Default `11434`.
- `--partition`, `--nodes`, `--gpus-per-node`: Written into the `#SBATCH` lines. Defaults `hcc`, `1`, `1`.
- `--node-domain`: Domain appended to the compute node name for the tunnel. Default `tugraz.at`.
- `--ready-timeout`: Seconds to wait for `GET /api/version` through the tunnel. Default `600`.
- `--port-forwarding`: Enable port forwarding.
- `--stop` (optional): Stop the SBATCH job after the script has finished.
- `--keep-alive`: Keep the job running until Ctrl+C.

## Job monitoring

`squeue -h -o "%i|%u|%T|%M|%N"` is polled every 10 seconds until the job state is `RUNNING`. Monitoring
fails (and the job is cancelled) if the job reaches a terminal state (`FAILED`, `CANCELLED`, `TIMEOUT`, ...)
or is not found in the queue six times in a row.

## SSH Port Forwarding

If `--port-forwarding` is enabled, `localhost:<local-ollama-port>` is forwarded to `127.0.0.1:<ollama-port>` on the
compute node. The tunnel runs in a background thread and is restarted automatically if it drops. The script is only
started once the Ollama API answers through the tunnel.

## Error Handling and Interrupts

On Ctrl+C or SIGTERM (`docker stop`), the scheduler stops the port forwarding and cancels the job with `scancel`.
A job that was submitted but never became usable (monitoring or readiness check failed) is always cancelled.

Exit codes: the script's exit code; `1` for scheduler errors (submission, monitoring, readiness); `130` when interrupted.

## SBATCH template (`ollama_serve.sh`)

Logs go to `ollama_serve_<jobid>.log` / `.err`, together with the Ollama version and `nvidia-smi` GPU info.
Ollama listens on `127.0.0.1` on the node and is only reachable through the tunnel.
