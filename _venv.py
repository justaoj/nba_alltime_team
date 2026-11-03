"""Make a script run inside this project's .venv, whichever Python started it.
If .venv exists and the current interpreter is not it, re-run the same command with the .venv Python.
Used by 01_download.py and run_all.py, so the VS Code Run button works with any selected interpreter."""
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
VENV = ROOT / ".venv"
VENV_PY = VENV / ("Scripts/python.exe" if os.name == "nt" else "bin/python3")


def ensure():
    if not VENV_PY.exists():
        sys.exit("No .venv yet: run 00_setup.py first (open it in VS Code and press the Run button).")
    if Path(sys.prefix).resolve() != VENV.resolve():
        os.chdir(ROOT)
        sys.exit(subprocess.call([str(VENV_PY), *sys.argv]))
    os.chdir(ROOT)
