"""Run every analysis step in order (02-11). Step 01 (download) runs separately because it takes ~45 minutes.
Run:  python3 run_all.py              (Windows, inside the project venv: .venv\\Scripts\\python.exe run_all.py)
      python3 run_all.py --download   (download first, then everything else)
"""
import _venv
_venv.ensure()                      # re-runs this script inside .venv if needed
import subprocess
import sys
import time
import config

STEPS = [
    ("02_parse.py", "parse pages into tables"),
    ("03_primes.py", "player primes and the all-player ranking (Fig. 5)"),
    ("04_metric_test.py", "metric accuracy: retrodiction test (Fig. 2)"),
    ("05_metric_awards.py", "metric vs MVP / All-NBA (Fig. 3)"),
    ("06_case_jordan.py", "case study: Michael Jordan 1987-90 (Table II, Fig. 4)"),
    ("07_team_model.py", "team models + cross-validation (Fig. 6)"),
    ("08_calibrate.py", "calibrate the game simulator"),
    ("09_search.py", "search all position-valid fives"),
    ("10_simulate.py", "Monte Carlo: dynasties, league season, playoffs (Fig. 7, 8)"),
    ("11_paper_assets.py", "tables, numbers and figures for the paper"),
]

steps = list(STEPS)
if "--download" in sys.argv:
    steps.insert(0, ("01_download.py", "download Basketball-Reference pages"))
if not any(config.RAW.glob("NBA_*.html")) and not (config.DATA / "games.csv").exists():
    sys.exit("No data yet: run 01_download.py first (or: run_all.py --download).")

t_all = time.time()
for script, what in steps:
    print(f"\n===== {script}: {what} =====", flush=True)
    t = time.time()
    r = subprocess.run([sys.executable, script])
    if r.returncode != 0:
        sys.exit(f"{script} failed; fix it and re-run from that step: {sys.executable} {script}")
    print(f"({time.time() - t:.0f} s)", flush=True)
print(f"\nAll steps done in {time.time() - t_all:.0f} s. Figures: figures/, results: results/, paper assets: paper/")
