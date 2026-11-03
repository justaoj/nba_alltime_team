"""Step 1: download Basketball-Reference season pages for 1979-80 to 2025-26.

Pages per season:
  NBA_{year}.html            league page (team pace, ORtg, DRtg, net rating)
  NBA_{year}_advanced.html   player advanced stats (BPM, USG%, TS%, ...)
  NBA_{year}_per_poss.html   player per-100-possession stats (ORtg, DRtg, ...)
  NBA_{year}_per_game.html   player per-game stats (for the 82-0 game's formula)
  NBA_{year}_ratings.html    team ratings (ORtg, DRtg, net rating)
  NBA_{year}_games-*.html    game results, one page per month

Every page is cached in data/raw/. Re-running skips pages already downloaded,
so you can stop with Ctrl+C and resume later.
Run:  python3 01_download.py            (all seasons, about 45 minutes the first time)
      python3 01_download.py 2026       (one season)
"""
import _venv
_venv.ensure()                      # re-runs this script inside .venv if needed
import re
import sys
import time
import requests
import config

BASE = "https://www.basketball-reference.com/leagues"
session = requests.Session()
session.headers["User-Agent"] = config.USER_AGENT


TRIES = 6


def fetch(name):
    """Download one page into data/raw unless it is cached. Returns the HTML text.
    Timeouts, dropped connections and server errors are retried with a growing pause (30 s, 60 s, ...)."""
    path = config.RAW / name
    if path.exists() and path.stat().st_size > 50_000:
        return path.read_text(encoding="utf-8")
    for attempt in range(1, TRIES + 1):
        try:
            r = session.get(f"{BASE}/{name}", timeout=(15, 90))   # 15 s to connect, 90 s to read
        except requests.RequestException as e:
            wait = 30 * attempt
            print(f"  {name}: {type(e).__name__}, retry {attempt}/{TRIES} in {wait} s", flush=True)
            time.sleep(wait)
            continue
        time.sleep(config.SLEEP_SECONDS)
        if r.status_code == 200:
            r.encoding = "utf-8"          # the site sends no charset; requests would guess ISO-8859-1
            path.write_text(r.text, encoding="utf-8")
            print(f"  got {name}", flush=True)
            return r.text
        if r.status_code == 429:
            sys.exit("HTTP 429: rate-limited. Wait an hour, then run 01_download.py again (it resumes).")
        if r.status_code == 404:
            return None
        wait = 30 * attempt
        print(f"  {name}: HTTP {r.status_code}, retry {attempt}/{TRIES} in {wait} s", flush=True)
        time.sleep(wait)
    sys.exit(f"could not download {name} after {TRIES} tries. Check the internet connection and run "
             "01_download.py again; it skips the pages already saved.")


def main():
    config.RAW.mkdir(parents=True, exist_ok=True)
    years = [int(a) for a in sys.argv[1:]] or range(config.FIRST_SEASON, config.LAST_SEASON + 1)
    for year in years:
        print(year, flush=True)
        fetch(f"NBA_{year}.html")          # league page: team pace, ORtg, DRtg, net rating
        for kind in ["advanced", "per_poss", "per_game", "ratings"]:
            fetch(f"NBA_{year}_{kind}.html")
        # the season's schedule page links to one page per month
        first = fetch(f"NBA_{year}_games.html")
        # month pages: games-november.html, and in 2019-20 games-october-2019.html / games-october-2020.html
        months = sorted(set(re.findall(rf"/leagues/NBA_{year}_games-([a-z]+(?:-\d{{4}})?)\.html", first or "")))
        for m in months:
            fetch(f"NBA_{year}_games-{m}.html")


if __name__ == "__main__":
    main()
