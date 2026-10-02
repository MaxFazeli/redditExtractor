import argparse
import os
import re
import signal
import subprocess
import sys
import threading
import time
import urllib.request

from utils import parse_remote_host, ssh_prefix
from check_squeue import check_squeue, TERMINAL_STATES

EXIT_SCHEDULER_ERROR = 1

# State shared with the signal handler / cleanup
port_forwarding_process = None
stop_port_forwarding = threading.Event()


def log(msg):
    print(f"[sbatch_scheduler {time.strftime('%Y-%m-%d %H:%M:%S')}] {msg}", flush=True)


def port_number(value):
    port = int(value)
    if not 1024 <= port <= 65535:
        raise argparse.ArgumentTypeError(f"{value} is not a valid port (1024-65535)")
    return port


def get_remote_home_directory(host):
    try:
        result = subprocess.run(ssh_prefix(host) + ['echo $HOME'], capture_output=True, text=True, check=True, timeout=60)
        return result.stdout.strip() or None
    except Exception as e:
        log(f"An error occurred while retrieving the home directory: {e}")
        return None


def replace_parameter_in_script(script_content, parameter_name, new_value):
    """Replace "<parameter_name>=<value>" in the script (e.g. PORT_NUMBER=11434, --gpus=1)."""
    pattern = rf'{re.escape(parameter_name)}=\S+'
    return re.sub(pattern, f'{parameter_name}={new_value}', script_content)


def create_script_on_remote(script_content, remote_path, host, skip_creation):
    try:
        check = subprocess.run(ssh_prefix(host) + [f'test -f {remote_path} && echo exists || echo not exists'],
                               capture_output=True, text=True, timeout=60)
        if check.stdout.strip() == "exists":
            if skip_creation:
                log("Skip creation: SBATCH script already exists on the remote host.")
                return True
            log("Re-creating SBATCH script on remote host...")

        process = subprocess.run(ssh_prefix(host) + [f'cat > {remote_path}'],
                                 input=script_content.encode(), capture_output=True, timeout=60)
        if process.returncode == 0:
            log(f"Successfully created script at {remote_path} on remote host")
            return True
        log(f"Failed to create the script on the remote host: {process.stderr.decode().strip()}")
        return False

    except Exception as e:
        log(f"An error occurred while creating the script on the remote host: {e}")
        return False


def submit_job(sbatch_script, host, remote_dir=None):
    """Submit the job (from remote_dir, so relative log paths in the script end up there)."""
    try:
        remote_command = f'sbatch {sbatch_script}'
        if remote_dir:
            remote_command = f'cd {remote_dir} && {remote_command}'
        command = ssh_prefix(host) + [remote_command] if host else ['sbatch', sbatch_script]
        result = subprocess.run(command, capture_output=True, text=True, check=True, timeout=120)

        for line in result.stdout.strip().split('\n'):
            if "Submitted batch job" in line:
                job_id = int(line.split()[-1])
                log(f"Submitted batch job {job_id}")
                return job_id

        log(f"Failed to parse job ID from submission output: {result.stdout} {result.stderr}")
        return None

    except subprocess.CalledProcessError as e:
        log(f"sbatch failed: {e.stderr.strip()}")
        return None
    except Exception as e:
        log(f"An error occurred while submitting job: {e}")
        return None


def monitor_job(job_id, user, host, max_not_found=6, poll_interval=10):
    """Wait until the job is RUNNING. Returns (True, node) or (False, None).

    Fails if the job reaches a terminal state, or is not found in the queue several times in a row
    (e.g. because it failed immediately) - previously this case looped forever.
    """
    not_found = 0
    while True:
        job = check_squeue(jobid=job_id, user=user, remote_host=host)
        if job is None:
            not_found += 1
            if not_found >= max_not_found:
                log(f"Job {job_id} is not in the queue anymore - it probably failed. "
                    f"Check ollama_serve_{job_id}.log/.err on the cluster.")
                return False, None
            log(f"Job {job_id} not found ({not_found}/{max_not_found}), retrying in {poll_interval} seconds...")
            time.sleep(poll_interval)
            continue

        not_found = 0
        if job["state"] == "RUNNING":
            log(f"Job {job_id} is running for USER {job['user']} on node {job['node']} (Time: {job['time']})")
            return True, job["node"]
        if job["state"] in TERMINAL_STATES:
            log(f"Job {job_id} ended with state {job['state']} before running.")
            return False, None

        log(f"Job {job_id} is {job['state']} ({job['node']}). Checking again in {poll_interval} seconds...")
        time.sleep(poll_interval)


def is_job_alive(job_id, user, host):
    job = check_squeue(jobid=job_id, user=user, remote_host=host)
    return job is not None and job["state"] not in TERMINAL_STATES


def stop_job(job_id, host):
    if job_id is None:
        return
    try:
        command = ssh_prefix(host) + ['scancel', str(job_id)] if host else ['scancel', str(job_id)]
        result = subprocess.run(command, capture_output=True, text=True, timeout=60)
        if result.returncode == 0:
            log(f"Successfully canceled job {job_id}")
        else:
            log(f"Failed to cancel job {job_id}: {result.stderr.strip()}")
    except Exception as e:
        log(f"An error occurred while stopping the job: {e}")


def execute_port_forwarding(local_port, remote_port, user, node, node_domain):
    """Start an SSH tunnel localhost:<local_port> -> <node>:<remote_port> in a background thread.

    The tunnel is restarted automatically if the SSH connection drops while the job is running.
    """
    target = f'{user}@{node}.{node_domain}' if node_domain else f'{user}@{node}'

    def port_forward():
        global port_forwarding_process
        while not stop_port_forwarding.is_set():
            cmd = [
                'ssh',
                '-o', 'BatchMode=yes',
                '-o', 'StrictHostKeyChecking=no',
                '-o', 'UserKnownHostsFile=/dev/null',
                '-o', 'LogLevel=ERROR',
                '-o', 'ExitOnForwardFailure=yes',
                '-o', 'TCPKeepAlive=yes',
                '-o', 'ServerAliveInterval=30',
                '-o', 'ServerAliveCountMax=3',
                '-L', f'{local_port}:127.0.0.1:{remote_port}',
                '-N',
                target
            ]
            try:
                port_forwarding_process = subprocess.Popen(cmd)
                return_code = port_forwarding_process.wait()
            except Exception as e:
                log(f"An error occurred while executing port forwarding: {e}")
                return_code = -1
            if not stop_port_forwarding.is_set():
                log(f"Port forwarding to {target} exited (code {return_code}); restarting in 10 seconds...")
                stop_port_forwarding.wait(10)

    log(f"Port forwarding localhost:{local_port} -> {target}:{remote_port}")
    thread = threading.Thread(target=port_forward, daemon=True)
    thread.start()
    return thread


def wait_for_ollama(local_port, job_id, user, host, timeout):
    """Wait until the Ollama API answers through the tunnel. Returns the server version or None."""
    url = f'http://127.0.0.1:{local_port}/api/version'
    deadline = time.monotonic() + timeout
    last_job_check = 0.0
    while time.monotonic() < deadline:
        try:
            with urllib.request.urlopen(url, timeout=5) as response:
                version = response.read().decode().strip()
                log(f"Ollama is ready: {version}")
                return version
        except Exception:
            pass
        if time.monotonic() - last_job_check > 30:
            last_job_check = time.monotonic()
            if not is_job_alive(job_id, user, host):
                log(f"Job {job_id} ended while waiting for Ollama to start.")
                return None
        time.sleep(3)
    log(f"Ollama did not become reachable on {url} within {timeout} seconds.")
    return None


def cleanup(job_id, host, stop=True):
    """Terminate port forwarding and (optionally) cancel the job."""
    stop_port_forwarding.set()
    if port_forwarding_process and port_forwarding_process.poll() is None:
        port_forwarding_process.terminate()
        log("Port forwarding stopped.")
    if stop:
        stop_job(job_id, host)


def run_script(file):
    """Run the given script and return its exit code."""
    _, extension = os.path.splitext(file)
    if extension == '.py':
        log(f"Running the Python script: {file}")
        return subprocess.run([sys.executable, file]).returncode
    if extension == '.sh':
        log(f"Running the Shell script: {file}")
        return subprocess.run(['bash', file]).returncode
    log(f"Unsupported file extension: {extension}")
    return EXIT_SCHEDULER_ERROR


def _raise_keyboard_interrupt(signum, frame):
    raise KeyboardInterrupt(f"signal {signum}")


def main():
    parser = argparse.ArgumentParser(description="Submit and monitor SBATCH job on a remote host, then execute a script locally when the job starts running, optionally stopping the job after completion.")
    parser.add_argument('--sbatch-script', type=str, help="The local SBATCH script file to copy and submit.")
    parser.add_argument('--remote-sbatch-script', type=str, help="The SBATCH script path to use on the remote host if it exists.")
    parser.add_argument('--sbatch-script-dir', type=str, help="The directory on the remote host where the SBATCH script should be created. Defaults to the user's home directory.")
    parser.add_argument('--skip-sbatch-script-creation', action='store_true', help="Skip the creation of the SBATCH script on the remote host if it already exists.")
    parser.add_argument('--ollama-port', type=port_number, default=11434, help="The PORT_NUMBER to be used in the SBATCH script (must be unique per running instance).")
    parser.add_argument('--partition', type=str, default="hcc", help="The partition to be used in the SBATCH script.")
    parser.add_argument('--nodes', type=int, default=1, help="The number of nodes to be used in the SBATCH script.")
    parser.add_argument('--gpus-per-node', type=int, default=1, help="The number of gpus per node to be used in the SBATCH script.")
    parser.add_argument('--local-ollama-port', type=port_number, default=11434, help="The local port for port forwarding.")
    parser.add_argument('--node-domain', type=str, default="tugraz.at", help="Domain appended to the compute node name for the SSH tunnel.")
    parser.add_argument('--ready-timeout', type=int, default=600, help="Seconds to wait for the Ollama API to answer through the tunnel.")
    parser.add_argument('--script', type=str, help="The Python/shell script to run locally when the job starts running.")
    parser.add_argument('-u', '--user', type=str, required=True, help="Specify USER to filter.")
    parser.add_argument('-host', '--host', type=str, required=True, help="Specify the remote HOST in format user@host:port to submit and monitor the job.")
    parser.add_argument('--stop', action='store_true', help="Stop the specific SBATCH job after running the script.")
    parser.add_argument('--keep-alive', action='store_true', help="Keep the job running until the script is manually stopped using Ctrl+C.")
    parser.add_argument('--port-forwarding', action='store_true', help="Enable port forwarding.")

    args = parser.parse_args()

    if not (args.sbatch_script or args.remote_sbatch_script):
        log("You must specify either --sbatch-script or --remote-sbatch-script.")
        return EXIT_SCHEDULER_ERROR

    # `docker stop` sends SIGTERM (not SIGINT): treat it like Ctrl+C so the SLURM job is always cancelled
    # and GPUs are not left allocated.
    signal.signal(signal.SIGTERM, _raise_keyboard_interrupt)
    if hasattr(signal, 'SIGHUP'):
        signal.signal(signal.SIGHUP, _raise_keyboard_interrupt)

    job_id = None
    exit_code = EXIT_SCHEDULER_ERROR
    should_stop = args.stop or args.keep_alive
    launched = False  # becomes True once the job runs and Ollama is reachable
    try:
        if args.sbatch_script:
            with open(args.sbatch_script, 'r') as file:
                original_script_content = file.read()
            script_content = replace_parameter_in_script(original_script_content, 'PORT_NUMBER', args.ollama_port)
            script_content = replace_parameter_in_script(script_content, '--partition', args.partition)
            script_content = replace_parameter_in_script(script_content, '--nodes', args.nodes)
            script_content = replace_parameter_in_script(script_content, '--gpus-per-node', args.gpus_per_node)
            script_content = replace_parameter_in_script(script_content, '--gpus', args.gpus_per_node)

            remote_dir = args.sbatch_script_dir or get_remote_home_directory(args.host)
            if not remote_dir:
                log("Unable to retrieve the remote home directory.")
                return EXIT_SCHEDULER_ERROR
            # One script per port, so that several instances do not overwrite each other's script.
            name, extension = os.path.splitext(os.path.basename(args.sbatch_script))
            sbatch_script = f"{remote_dir}/{name}_{args.ollama_port}{extension}"
            if not create_script_on_remote(script_content, sbatch_script, args.host, args.skip_sbatch_script_creation):
                log("Script creation failed. Exiting.")
                return EXIT_SCHEDULER_ERROR
        else:
            sbatch_script = args.remote_sbatch_script
            remote_dir = os.path.dirname(sbatch_script) or None

        job_id = submit_job(sbatch_script, args.host, remote_dir)
        if job_id is None:
            return EXIT_SCHEDULER_ERROR

        success, node = monitor_job(job_id, args.user, args.host)
        if not success:
            log(f"Error running job ({job_id})")
            return EXIT_SCHEDULER_ERROR

        log(f"Job ({job_id}) running on node '{node}'")
        if args.port_forwarding:
            execute_port_forwarding(args.local_ollama_port, args.ollama_port, args.user, node, args.node_domain)
            if wait_for_ollama(args.local_ollama_port, job_id, args.user, args.host, args.ready_timeout) is None:
                return EXIT_SCHEDULER_ERROR
        else:
            target = f"{args.user}@{node}.{args.node_domain}" if args.node_domain else f"{args.user}@{node}"
            log(f"To enable port forwarding, run the following command:\nssh -o TCPKeepAlive=yes -o ServerAliveInterval=30 -L {args.local_ollama_port}:127.0.0.1:{args.ollama_port} {target}")

        launched = True
        exit_code = 0
        if args.script:
            exit_code = run_script(args.script)
            log(f"Script finished with exit code {exit_code}")

        if args.keep_alive:
            log("Keep-alive mode enabled. Press Ctrl+C to stop the job and exit.")
            while True:
                time.sleep(1)

    except KeyboardInterrupt as e:
        log(f"Program interrupted ({e or 'Ctrl+C'}). Cleaning up...")
        exit_code = 130
        should_stop = True
    finally:
        if job_id is not None:
            # A job that was submitted but never became usable is always cancelled.
            should_stop = should_stop or not launched
            if should_stop:
                log(f"Stopping job ID: {job_id}")
            cleanup(job_id, args.host, stop=should_stop)
        elif port_forwarding_process:
            cleanup(None, args.host, stop=False)

    return exit_code


if __name__ == "__main__":
    sys.exit(main())
