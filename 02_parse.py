"""Step 2: parse the cached pages into three CSV tables in data/.

  player_seasons.csv  one row per player x season x team (traded players also get a 2TM/3TM total row);
                      includes the awards column (e.g. "MVP-1,AS,NBA1") used to validate the metric
  team_seasons.csv    one row per team x season: conference, pace, ORtg, DRtg, net rating, wins, Four Factors
  games.csv           every regular-season game: date, home, away, points, margin

Cleaning rules:
  - traded players: keep both the per-team rows and the multi-team total (agg=True);
    later steps use the total for player primes and the per-team rows for team models
  - regular season only: the first N games of the season in date order, N = sum of team games / 2;
    the NBA Cup final (called the In-Season Tournament in 2023-24) is dropped first because it does not count in the standings
Run:  python3 02_parse.py
"""
import re
import sys
import pandas as pd
import config
from common import read_table, to_num

ADV = ["pos", "age", "games", "games_started", "mp", "per", "ts_pct", "fg3a_per_fga_pct", "fta_per_fga_pct",
       "orb_pct", "drb_pct", "trb_pct", "ast_pct", "stl_pct", "blk_pct", "tov_pct", "usg_pct",
       "ows", "dws", "ws", "ws_per_48", "obpm", "dbpm", "bpm", "vorp", "awards"]
PP = ["off_rtg", "def_rtg", "fg3_pct", "ft_pct", "efg_pct", "fg3a_per_poss", "pts_per_poss"]
PG = ["mp_per_g", "pts_per_g", "trb_per_g", "ast_per_g", "stl_per_g", "blk_per_g"]
MONTHS = ["july", "august", "september", "october", "november", "december", "january", "february",
          "march", "april", "may", "june"]


def player_table(year, kind, table_id, fields):
    rows = read_table(config.RAW / f"NBA_{year}_{kind}.html", table_id)
    out = []
    for r in rows:
        if "name_display_id" not in r:
            continue
        rec = {"player_id": r["name_display_id"], "name": r["name_display"], "team": r.get("team_name_abbr", "")}
        rec.update({f: r.get(f) for f in fields})
        out.append(rec)
    d = pd.DataFrame(out)
    d["k"] = d.groupby(["player_id", "team"]).cumcount()
    return d


def players(year):
    a = player_table(year, "advanced", "advanced", ADV)
    p = player_table(year, "per_poss", "per_poss", PP).drop(columns="name")
    g = player_table(year, "per_game", "per_game_stats", PG).drop(columns=["name"])
    m = a.merge(p, on=["player_id", "team", "k"], how="left").merge(g, on=["player_id", "team", "k"], how="left")
    m = m.drop(columns="k")
    m["season"] = year
    return m


def teams(year):
    rows = read_table(config.RAW / f"NBA_{year}.html", "advanced-team")
    out = []
    for r in rows:
        if "team_abbr" not in r:
            continue
        out.append({"season": year, "team": r["team_abbr"], "team_name": r["team"].rstrip("*"),
                    "wins": r["wins"], "losses": r["losses"], "mov": r["mov"], "srs": r["srs"],
                    "pace": r["pace"], "ortg": r["off_rtg"], "drtg": r["def_rtg"], "nrtg": r.get("net_rtg"),
                    "efg": r["efg_pct"], "tov": r["tov_pct"], "orb": r["orb_pct"], "ftr": r["ft_rate"],
                    "opp_efg": r["opp_efg_pct"], "opp_tov": r["opp_tov_pct"], "drb": r["drb_pct"],
                    "opp_ftr": r["opp_ft_rate"]})
    T = pd.DataFrame(out)
    # conference (E/W) from the team ratings page, used to seed the playoff simulation
    conf = {r["team_name_abbr"]: r.get("conf_id") for r in read_table(config.RAW / f"NBA_{year}_ratings.html", "ratings")
            if "team_name_abbr" in r}
    T["conf"] = T.team.map(conf)
    return T


def games(year, team_games):
    files = sorted(config.RAW.glob(f"NBA_{year}_games-*.html"))     # order does not matter: sorted by date below
    out = []
    for f in files:
        for r in read_table(f, "schedule"):
            if not r.get("home_pts") or not r.get("visitor_pts"):
                continue
            out.append({"season": year, "date": pd.to_datetime(r["date_game"]),
                        "home": r.get("home_team_name_abbr"), "away": r.get("visitor_team_name_abbr"),
                        "home_pts": int(r["home_pts"]), "away_pts": int(r["visitor_pts"]),
                        "remark": r.get("game_remarks", "")})
    g = pd.DataFrame(out)
    g = g.sort_values("date", kind="stable").reset_index(drop=True)
    cup = g[g.remark.str.contains("Cup|In-Season Tournament", na=False)]
    if year >= 2024 and len(cup):
        g = g.drop(cup.date.idxmax())          # NBA Cup final: not in the standings
    n_regular = team_games // 2
    g = g.iloc[:n_regular].copy()
    g["margin"] = g.home_pts - g.away_pts
    return g.drop(columns="remark")


def complete(y):
    need = [f"NBA_{y}.html"] + [f"NBA_{y}_{k}.html" for k in ["advanced", "per_poss", "per_game", "ratings"]]
    return all((config.RAW / n).exists() for n in need) and any(config.RAW.glob(f"NBA_{y}_games-*.html"))


def main():
    years = [y for y in range(config.FIRST_SEASON, config.LAST_SEASON + 1) if complete(y)]
    missing = sorted(set(range(config.FIRST_SEASON, config.LAST_SEASON + 1)) - set(years))
    if not years:
        sys.exit("no downloaded seasons in data/raw; run 01_download.py first (the tables in data/ were left as they are)")
    if missing:
        print("skipping seasons with missing pages (run 01_download.py):", missing)
    P = pd.concat([players(y) for y in years], ignore_index=True)
    P = to_num(P, skip=("player_id", "name", "team", "pos", "awards"))
    P["agg"] = P.team.str.fullmatch(r"\dTM|TOT")
    P.to_csv(config.DATA / "player_seasons.csv", index=False)
    print("player_seasons", P.shape)

    T = pd.concat([teams(y) for y in years], ignore_index=True)
    T = to_num(T, skip=("team", "team_name", "conf"))
    T["nrtg"] = T.ortg - T.drtg
    T.to_csv(config.DATA / "team_seasons.csv", index=False)
    print("team_seasons", T.shape)

    G = []
    for y in years:
        tg = int((T[T.season == y].wins + T[T.season == y].losses).sum())
        g = games(y, tg)
        counts = pd.concat([g.home, g.away]).value_counts()
        check = T[T.season == y].set_index("team")
        bad = (counts.reindex(check.index) != check.wins + check.losses).sum()
        if bad:
            print(f"  warning {y}: {bad} teams with a game count that does not match W+L")
        G.append(g)
    G = pd.concat(G, ignore_index=True)
    G.to_csv(config.DATA / "games.csv", index=False)
    print("games", G.shape)


if __name__ == "__main__":
    main()
