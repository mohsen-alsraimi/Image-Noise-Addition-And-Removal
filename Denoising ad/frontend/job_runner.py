"""
Starts one experiment in a separate Python process and watches it.

The experiment itself is run by the normal backend:

    python main.py --job <job.json>

Job files and logs are kept in the local .frontend_jobs folder.
"""

import os
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

from experiment_job import (
    ACTIVE_FILE_NAME,
    ACTIVE_STATES,
    PROJECT_DIR,
    mark_job,
    read_json,
    validate_job_filters,
    write_json,
)

JOBS_DIR = PROJECT_DIR / ".frontend_jobs"
START_GRACE_SECONDS = 30


class JobAlreadyRunning(RuntimeError):
    """Raised when a second experiment is started too early."""


def is_process_running(pid):
    if os.name != "nt":
        try:
            os.kill(pid, 0)
            return True
        except OSError:
            return False

    # On Windows, os.kill(pid, 0) is unsafe, so ask the system directly.
    import ctypes

    SYNCHRONIZE = 0x00100000
    WAIT_TIMEOUT = 0x00000102
    kernel32 = ctypes.windll.kernel32
    handle = kernel32.OpenProcess(SYNCHRONIZE, False, pid)
    if not handle:
        return False

    running = kernel32.WaitForSingleObject(handle, 0) == WAIT_TIMEOUT
    kernel32.CloseHandle(handle)
    return running


def _process_may_be_running(job_dir):
    pid_file = job_dir / "pid.txt"

    try:
        if pid_file.is_file():
            return is_process_running(int(pid_file.read_text(encoding="utf-8")))

        # The process may be starting and its id may not be saved yet.
        age = time.time() - (job_dir / "status.json").stat().st_mtime
        return age < START_GRACE_SECONDS
    except (OSError, ValueError):
        return False


def job_state(job_id):
    """Returns the saved state, and detects a process that stopped early."""
    job_dir = JOBS_DIR / job_id
    status = read_json(job_dir / "status.json") or {}

    if status.get("state") in (*ACTIVE_STATES, None):
        if not _process_may_be_running(job_dir):
            # Read again, because the job may have finished a moment ago.
            status = read_json(job_dir / "status.json") or {}

            if status.get("state") in (*ACTIVE_STATES, None):
                job_dir.mkdir(parents=True, exist_ok=True)
                mark_job(
                    job_dir,
                    "failed",
                    error="The experiment process stopped before finishing.",
                )
                status = read_json(job_dir / "status.json") or {}

    if not status:
        return {"state": "starting"}

    return status


def active_job_id():
    """Returns the running job id, or None."""
    active_file = JOBS_DIR / ACTIVE_FILE_NAME

    try:
        job_id = active_file.read_text(encoding="utf-8").strip()
        age = time.time() - active_file.stat().st_mtime
    except OSError:
        return None

    if not job_id:
        if age > START_GRACE_SECONDS:
            active_file.unlink(missing_ok=True)
        return None

    return job_id if job_state(job_id)["state"] in ACTIVE_STATES else None


def start_job(job):
    """Saves a job file and starts main.py. Only one job can run at a time."""
    validate_job_filters(job)
    JOBS_DIR.mkdir(parents=True, exist_ok=True)

    ignore_file = JOBS_DIR / ".gitignore"
    if not ignore_file.exists():
        ignore_file.write_text("*\n", encoding="utf-8")

    if active_job_id() is not None:
        raise JobAlreadyRunning("An experiment is already running.")

    job_id = datetime.now().strftime("job_%Y%m%d_%H%M%S_%f")

    try:
        with open(JOBS_DIR / ACTIVE_FILE_NAME, "x", encoding="utf-8") as active_file:
            active_file.write(job_id)
    except FileExistsError as error:
        raise JobAlreadyRunning("An experiment is already running.") from error

    job_dir = JOBS_DIR / job_id
    job_dir.mkdir()
    write_json(job_dir / "job.json", job)
    mark_job(job_dir, "starting")

    environment = dict(os.environ, MPLBACKEND="Agg", PYTHONIOENCODING="utf-8")
    no_window = subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0

    try:
        with open(job_dir / "run.log", "w", encoding="utf-8") as log:
            process = subprocess.Popen(
                [sys.executable, "-B", "-u", str(PROJECT_DIR / "main.py"),
                 "--job", str(job_dir / "job.json")],
                cwd=PROJECT_DIR,
                env=environment,
                stdin=subprocess.DEVNULL,
                stdout=log,
                stderr=subprocess.STDOUT,
                creationflags=no_window,
            )
    except OSError as error:
        mark_job(job_dir, "failed", error=f"Could not start the experiment: {error}")
        raise

    (job_dir / "pid.txt").write_text(str(process.pid), encoding="utf-8")
    return job_id


def result_is_complete(result_name, results_dir=None):
    """True only if the saved run_config.json says the run completed."""
    if not result_name:
        return False

    folder = Path(results_dir or PROJECT_DIR / "results") / result_name
    config = read_json(folder / "run_config.json") or {}
    return config.get("run", {}).get("status") == "completed"


def read_log_tail(job_id, lines=25):
    """Returns the end of the job log for failure messages."""
    try:
        text = (JOBS_DIR / job_id / "run.log").read_text(
            encoding="utf-8", errors="replace"
        )
    except OSError:
        return ""

    return "\n".join(text.splitlines()[-lines:])
