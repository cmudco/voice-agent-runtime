"""Run any local backend under bounded, metadata-only diagnostic capture.

python scripts/local_logs.py --service sb-backend --directory local-logs/run-1 -- <command>
Run from the service's working directory; the child inherits it. No shell is used.
"""

import argparse
import os
import signal
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from runtime.local_diagnostics import make_log, write_line


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--service", choices=["runtime", "whisper", "sb-backend", "dare-backend"], required=True)
    parser.add_argument("--directory", required=True)
    parser.add_argument("command", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    command = args.command[1:] if args.command[:1] == ["--"] else args.command
    if not command:
        parser.error("A backend command is required after --")
    os.umask(0o077)
    log = make_log(args.service, args.directory)
    environment = {**os.environ, "PYTHONUNBUFFERED": "1"}
    with subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                          text=True, errors="replace", env=environment, start_new_session=True) as child:
        def forward(signum, _frame):
            if child.poll() is None:
                os.killpg(child.pid, signum)
        signal.signal(signal.SIGINT, forward)
        signal.signal(signal.SIGTERM, forward)
        for line in child.stdout:
            # Original output stays in the local terminal, not in the shareable file.
            print(line, end="", flush=True)
            write_line(log, line)
        return child.wait()


if __name__ == "__main__":
    sys.exit(main())
