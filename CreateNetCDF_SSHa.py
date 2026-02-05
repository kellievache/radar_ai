
#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Dispatch stack_to_netcdf jobs to multiple hosts over SSH.
- Uses the Python interpreter inside your working conda env directly
  (no 'conda activate' needed).
- Unsets PYTHONHOME/PYTHONPATH on the remote to avoid 'encodings' failures.
- --dry-run prints commands only (no SSH).
"""

import os
import sys
import shlex
import time
import threading
import argparse
from queue import Queue

import paramiko


# -----------------------------
# Defaults (override via CLI)
# -----------------------------
DEFAULT_USERNAME  = "vachek"
DEFAULT_KEY_FILE  = os.path.expanduser("/nfs/pancake/u2/home/vachek/.ssh/id_rsa" )

# The interpreter INSIDE the working env (this is the key change):
DEFAULT_ENV_PYTHON = "/nfs/pancake/u2/home/vachek/.conda/envs/AG_15_PY312/bin/python"

# Your CLI wrapper that runs ONE stack job:
DEFAULT_STACK_CLI = "/nfs/pancake/u5/projects/vachek/radar_ai/stack_to_netcdf.py"

DEFAULT_HOSTS = [
    "prismcpu.nacse.org",
    "prismgpu1.nacse.org",
    "prismcpu2.nacse.org",
    "prismcpu3.nacse.org",
    "prismcpu4.nacse.org",
]
alll=True
if alll:
    DEFAULT_TASKS = [
        # y train (WITH radar)
        dict(
            root="/nfs/pancake/u4/data/prism/us/an81/r1503/ehdr/800m/ppt/daily/",
            years="2002:2019",
            var_name="ppt",
            out_path="/nfs/pancake/u5/projects/vachek/radar_ai/netcdf/y_train.nc",
            include_text="adj_best_ppt",
        ),
        # y val (WITH radar)
        # dict(
        #     root="/nfs/pancake/u4/data/prism/us/an91/r2112/ehdr/800m/ppt/daily/",
        #     years="2023",
        #     var_name="ppt",
        #     out_path="/nfs/pancake/u5/projects/vachek/radar_ai/netcdf/y_val.nc",
        #     include_text="adj_best_ppt",
        # ),
        # x train (NO radar)
        dict(
            root="/nfs/pancake/u4/data/prism/us/an81/r1503/ehdr/800m/ppt/daily/",
            years="2002:2019",
            var_name="ppt",
            out_path="/nfs/pancake/u5/projects/vachek/radar_ai/netcdf/x_train.nc",
            include_text="cai_ppt",
        ),
        # x val (NO radar)
        # dict(
        #     root="/nfs/pancake/u4/data/prism/us/an91/r2112/ehdr/800m/ppt/daily/",
        #     years="2023",
        #     var_name="ppt",
        #     out_path="/nfs/pancake/u5/projects/vachek/radar_ai/netcdf/x_val.nc",
        #     include_text="cai_ppt",
        # ),
        # # infer-only (NO radar)
        # dict(
        #     root="/nfs/pancake/u4/data/prism/us/an91/r2112/ehdr/800m/ppt/daily/",
        #     years="2024",
        #     var_name="ppt",
        #     out_path="/nfs/pancake/u5/projects/vachek/radar_ai/netcdf/infer.nc",
        #     include_text="cai_ppt",
        # ),
        
        # dict(
        #     root="/nfs/pancake/u4/data/prism/us/an81/r1503/ehdr/800m/ppt/daily/",
        #     years="1982:1985",
        #     var_name="ppt",
        #     out_path="/nfs/pancake/u5/projects/vachek/radar_ai/netcdf/infer80_85.nc",
        #     include_text="cai_ppt",
        # ),
    ]
else:
    DEFAULT_TASKS = [
    
        dict(
            root="/nfs/pancake/prism_current/us/an/ehdr/800m/ppt/daily/normals/",
            years="2020",
            var_name="ppt",
            out_path="/nfs/pancake/u5/projects/vachek/radar_ai/netcdf/normals.nc",
            include_text="prism_ppt",
        ),
    ]

# -----------------------------
# CLI
# -----------------------------
def parse_args():
    ap = argparse.ArgumentParser(description="SSH dispatcher for stack_to_netcdf jobs.")
    ap.add_argument("--dry-run", action="store_true",
                    help="Print commands and exit without SSH.")
    ap.add_argument("--username", default=DEFAULT_USERNAME, help="SSH username")
    ap.add_argument("--key-file", default=DEFAULT_KEY_FILE, help="SSH private key file")
    ap.add_argument("--hosts", default=",".join(DEFAULT_HOSTS),
                    help="Comma-separated host list")
    ap.add_argument("--python", default=DEFAULT_ENV_PYTHON,
                    help="Absolute path to env's python (e.g., .../AG_15_PY312/bin/python)")
    ap.add_argument("--stack-cli", default=DEFAULT_STACK_CLI,
                    help="Path to stack_to_netcdf_cli.py on the remote filesystem")
    return ap.parse_args()


# -----------------------------
# SSH runner
# -----------------------------
def run_remote_command(host: str, username: str, key_filename: str, command: str, dry_run: bool = False):
    print(f"[{host}] COMMAND:\n{command}\n")
    if dry_run:
        print(f"[{host}] (dry-run) Skipping execution.\n")
        return

    ssh = paramiko.SSHClient()
    ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    ssh.connect(host, username=username, key_filename=key_filename)

    # A PTY makes the shell behave more like an interactive session (more consistent output)
    stdin, stdout, stderr = ssh.exec_command(command, get_pty=True)

    out = stdout.read().decode("utf-8", errors="replace")
    err = stderr.read().decode("utf-8", errors="replace")
    ssh.close()

    if out.strip():
        print(f"[{host}] STDOUT:\n{out}")
    if err.strip():
        print(f"[{host}] STDERR:\n{err}")


# -----------------------------
# Command builder
# -----------------------------

def build_remote_command_for_task(task: dict, *, python_path: str, stack_cli: str) -> str:
    """
    Build a robust remote command that:
      - Calls /bin/bash -lc with a single, fully-quoted payload,
      - Unsets PYTHONHOME/PYTHONPATH (avoid encodings crash),
      - Uses the env's python *by absolute path* (no activation needed),
      - Preflights 'import encodings' with single quotes,
      - Runs the CLI with safely quoted args.
    """
    def q(x: str) -> str:
        return shlex.quote(x)

    py  = q(python_path)
    cli = q(stack_cli)

    # Single-quoted Python -c payload (no nested double quotes)
    preflight = f"{py} -c 'import encodings,sys; print(sys.executable)'"

    # CLI invocation with safely quoted args
    py_cli = (
        f"{py} {cli} "
        f"--root {q(task['root'])} "
        f"--years {q(task['years'])} "
        f"--var_name {q(task['var_name'])} "
        f"--out_path {q(task['out_path'])} "
        f"--include_text {q(task['include_text'])}"
    )

    # Compose the bash payload first (no outer quotes yet)
    bash_payload = f"unset PYTHONHOME PYTHONPATH; {preflight} && {py_cli}"

    # Quote the entire payload once for the outer csh/tcsh to pass through intact
    # This yields: /bin/bash -lc 'unset ...; /env/bin/python -c '\''import ...'\'' && ...'
    return f"/bin/bash -lc {shlex.quote(bash_payload)}"


# -----------------------------
# Worker thread per host
# -----------------------------
def host_worker(host: str, q: Queue, username: str, key_file: str, python_path: str, stack_cli: str, dry_run: bool):
    while True:
        try:
            task = q.get_nowait()
        except Exception:
            break

        print(f"[{host}] Starting: out={task['out_path']}")
        cmd = build_remote_command_for_task(task, python_path=python_path, stack_cli=stack_cli)
        try:
            run_remote_command(host, username, key_file, cmd, dry_run=dry_run)
            print(f"[{host}] Finished: out={task['out_path']}\n")
        except Exception as e:
            print(f"[{host}] ERROR on {task['out_path']}: {e}\n")
        finally:
            q.task_done()
        time.sleep(0.5)


# -----------------------------
# Dispatcher
# -----------------------------
def dispatch_tasks_to_hosts(tasks, hosts, username, key_file, python_path, stack_cli, dry_run=False):
    tasks = list(tasks)
    hosts = [h.strip() for h in hosts if h.strip()]
    if not tasks:
        print("No tasks to run."); return
    if not hosts:
        print("No hosts supplied."); return

    # use as many hosts as tasks (extra hosts idle)
    hosts = hosts[:min(len(hosts), len(tasks))]

    q = Queue()
    for t in tasks:
        q.put(t)

    threads = []
    for h in hosts:
        print(f"Starting worker on {h}")
        th = threading.Thread(target=host_worker,
                              args=(h, q, username, key_file, python_path, stack_cli, dry_run),
                              daemon=True)
        th.start()
        threads.append(th)

    q.join()
    for i, th in enumerate(threads):
        th.join(timeout=0.1)
        print(f"Thread for {hosts[i]} complete.")
    print("All remote stacking tasks complete.")



# -----------------------------
# Main
# -----------------------------
def main():
    args = parse_args()

    hosts = args.hosts.split(",")

    # DRY-RUN sanity: show one constructed command per host (for the first few tasks)
    # The real run will still print exact commands per host.
    dispatch_tasks_to_hosts(
        tasks=DEFAULT_TASKS,
        hosts=hosts,
        username=args.username,
        key_file=args.key_file,
        python_path=args.python,      # absolute path to env's python
        stack_cli=args.stack_cli,
        dry_run=args.dry_run
    )


if __name__ == "__main__":
    main()
