"""Optional Manager hook: check existing dependencies, never run pip."""
from pathlib import Path
import subprocess
import sys

if __name__ == "__main__":
    raise SystemExit(subprocess.call([sys.executable, "-I", str(Path(__file__).with_name("worker.py")), "--check-deps"]))
