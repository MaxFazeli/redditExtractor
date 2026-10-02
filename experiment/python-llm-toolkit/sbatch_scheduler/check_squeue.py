import subprocess
from utils import parse_remote_host, ssh_prefix

# Fields requested from squeue: job id | user | state | elapsed time | node list (or pending reason)
SQUEUE_FORMAT = "%i|%u|%T|%M|%N"

# Job states after which the job will never run (again).
TERMINAL_STATES = {
    "BOOT_FAIL", "CANCELLED", "COMPLETED", "DEADLINE", "FAILED", "NODE_FAIL",
    "OUT_OF_MEMORY", "PREEMPTED", "SPECIAL_EXIT", "TIMEOUT",
}


def check_squeue(jobid=None, user=None, remote_host=None):
    """Query squeue for a job.

    Returns a dict {"jobid", "user", "state", "time", "node"} for the first matching job,
    or None if the job is not (or no longer) in the queue or squeue could not be queried.
    """
    command = ['squeue', '-h', '-o', f'"{SQUEUE_FORMAT}"']
    if user:
        command += ['-u', user]
    if jobid:
        command += ['-j', str(jobid)]

    if remote_host:
        command = ssh_prefix(remote_host) + command

    try:
        result = subprocess.run(command, capture_output=True, text=True, timeout=60)
    except subprocess.TimeoutExpired:
        print("squeue timed out")
        return None

    if result.returncode != 0:
        # squeue exits with an error ("Invalid job id specified") once a job has left the queue
        if result.stderr.strip():
            print(f"squeue: {result.stderr.strip()}")
        return None

    for line in result.stdout.strip().splitlines():
        parts = line.strip().strip('"').split('|')
        if len(parts) != 5:
            continue
        job_id, job_user, state, time_value, node = parts
        return {"jobid": job_id, "user": job_user, "state": state, "time": time_value, "node": node}
    return None


# Example usage
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Show the state of SLURM jobs")
    parser.add_argument('-j', '--jobid', type=int, help="Specify JOBID to filter")
    parser.add_argument('-u', '--user', type=str, help="Specify USER to filter")
    parser.add_argument('-host', '--host', type=str, help="Specify HOST in format user@host:port to run the squeue command")

    args = parser.parse_args()
    job = check_squeue(args.jobid, args.user, args.host)
    if job:
        print(f"JOB: {job['jobid']}, USER: {job['user']}, STATE: {job['state']}, TIME: {job['time']}, NODE: {job['node']}")
    else:
        print("Job not found or an error occurred.")
