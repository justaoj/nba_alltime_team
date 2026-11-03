"""Step 0: create this project's Python environment (.venv) and install the packages in requirements.txt.
Installs nothing outside this folder. Uses only the standard library, so any Python 3.10+ can run it.
Run:  python3 00_setup.py          (VS Code: open this file and press the Run button)
"""
import subprocess
import sys
import venv
from pathlib import Path

ROOT = Path(__file__).resolve().parent
VENV = ROOT / ".venv"
PY = VENV / ("Scripts/python.exe" if sys.platform == "win32" else "bin/python3")


def run(cmd):
    print(">", " ".join(str(c) for c in cmd), flush=True)
    r = subprocess.run([str(c) for c in cmd], cwd=ROOT)
    if r.returncode != 0:
        sys.exit(f"failed: {' '.join(str(c) for c in cmd)}")


def main():
    print(f"Python {sys.version.split()[0]} at {sys.executable}")
    if sys.version_info < (3, 10):
        sys.exit("Python 3.10 or newer is needed. Install it from https://www.python.org/downloads/")
    if not PY.exists():
        print(f"creating {VENV} ...", flush=True)
        venv.EnvBuilder(with_pip=True).create(VENV)
    run([PY, "-m", "pip", "install", "--upgrade", "pip", "--quiet"])
    run([PY, "-m", "pip", "install", "-r", ROOT / "requirements.txt"])
    run([PY, "-c", "import pandas, numpy, scipy, sklearn, matplotlib, bs4, lxml, requests; "
                   "print('OK: all packages installed in .venv')"])
    print("\nNext: open 01_download.py and press Run (about 45 minutes), then run_all.py.")


if __name__ == "__main__":
    main()
