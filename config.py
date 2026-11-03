"""Settings for the 'best NBA starting five of all time' project. Edit here, not inside the scripts."""
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent
RAW = ROOT / "data" / "raw"          # cached Basketball-Reference HTML pages (step 01)
DATA = ROOT / "data"                 # parsed CSV tables (step 02)
RESULTS = ROOT / "results"           # CSV / JSON output of every step
FIGURES = ROOT / "figures"           # PNG figures (copied into paper/figures by step 11)
PAPER = ROOT / "paper"

FIRST_SEASON = 1980                  # 1979-80: first season with the 3-point line and a full box score
LAST_SEASON = int(os.environ.get("LAST_SEASON", 2026))            # 2025-26: last completed season
TARGET_SEASON = int(os.environ.get("TARGET_SEASON", LAST_SEASON))  # league used in the season simulation

# Basketball-Reference blocks more than 20 requests a minute; 4 s between pages keeps us at 15.
SLEEP_SECONDS = 4.0
USER_AGENT = "Mozilla/5.0 (student research project; NBA best five)"

# Season length, used to scale the minutes threshold in short seasons.
SEASON_GAMES = {1999: 50, 2012: 66, 2020: 70, 2021: 72}
MIN_MINUTES_FULL_SEASON = 1500       # qualifying season = 1500+ minutes in an 82-game season

POSITIONS = ["PG", "SG", "SF", "PF", "C"]
# Lineup slots, as on All-NBA teams: two guards, two forwards, one center.
SLOTS = ["G", "G", "F", "F", "C"]
SLOT_GROUP = {"PG": "G", "SG": "G", "SF": "F", "PF": "F", "C": "C"}
PRIME_WINDOW = 3                     # seasons in a prime

# Case study (step 06): Michael Jordan's prime window, seasons named by the year they end.
CASE_PLAYER = "Michael Jordan"
CASE_SEASONS = [1988, 1989, 1990]    # 1987-88, 1988-89, 1989-90

# Simulation
N_SIMS = 100_000                     # Monte Carlo games / series / league seasons per matchup
SIM_CHUNK = 10_000                   # league seasons simulated at once (memory)
N_BOOTSTRAP = 200                    # resamples for uncertainty intervals
SEED = 2026

# Head-to-head opponents: famous dynasties (season = year it ends, Basketball-Reference team code).
# Played on a neutral court, single games and best-of-7 series.
DYNASTIES = [(1986, "BOS"), (1987, "LAL"), (1996, "CHI"), (2001, "LAL"), (2014, "SAS"), (2017, "GSW")]

# League simulation: the five joins the TARGET_SEASON league in place of this team
# ("worst" = lowest net rating that season), plays all regular-season games, the play-in and the playoffs.
REPLACE_TEAM = "worst"

# Candidate D: a consensus "fan" all-time five. Names exactly as on Basketball-Reference.
FAN_FIVE = ["Magic Johnson", "Michael Jordan", "LeBron James", "Tim Duncan", "Shaquille O'Neal"]

# The author's own pick before the analysis (step 13 simulates it next to the All-Time teams).
EXPECTED_FIVE = ["Stephen Curry", "Michael Jordan", "LeBron James", "Larry Bird", "David Robinson"]

for _p in (RAW, RESULTS, FIGURES, PAPER / "figures", PAPER / "tables"):
    _p.mkdir(parents=True, exist_ok=True)
