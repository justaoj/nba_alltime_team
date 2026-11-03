"""Step 6: case study - Michael Jordan's 3-season prime (1987-88 to 1989-90).

Shows the evaluation on one player before it is applied to everyone:
  - his minutes-weighted averages over the three seasons
  - the league-average player in the same seasons (minutes-weighted, so the average BPM is about 0)
  - his percentile among all qualifying player-seasons of those three seasons
  - his BPM rank in each season
Outputs: results/case_jordan.csv, results/case_jordan_seasons.csv, figures/fig4_jordan.png (paper Fig. 4)
Run:  python3 06_case_jordan.py
"""
import numpy as np
import pandas as pd
import config
from common import style, save, season_label, INK, INK2, MUTED, ACCENT, COL_W
import matplotlib.pyplot as plt

# (column, label, higher is better)
STATS = [("pts_per_g", "득점 PTS/G", True), ("ts_pct", "TS%", True), ("usg_pct", "USG%", True),
         ("ast_pct", "AST%", True), ("trb_pct", "TRB%", True), ("stl_pct", "STL%", True),
         ("blk_pct", "BLK%", True), ("tov_pct", "TOV% (낮을수록 좋음)", False), ("off_rtg", "ORtg", True),
         ("per", "PER", True), ("ws_per_48", "WS/48", True), ("obpm", "OBPM", True), ("dbpm", "DBPM", True),
         ("bpm", "BPM", True)]


def fmt(col, v):
    if col in ("ts_pct", "ws_per_48"):
        return f"{v:.3f}"
    if col in ("obpm", "dbpm", "bpm"):
        return f"{v:+.1f}"
    return f"{v:.1f}"


def main():
    style()
    S = pd.read_csv(config.RESULTS / "player_season_one.csv")
    seasons = config.CASE_SEASONS
    W = S[S.season.isin(seasons)]
    J = W[W.name == config.CASE_PLAYER]
    if len(J) != len(seasons):
        raise SystemExit(f"{config.CASE_PLAYER}: found {len(J)} of {len(seasons)} seasons; check CASE_PLAYER/CASE_SEASONS")
    Q = W[W.qualified]
    rows = []
    for col, lab, up in STATS:
        j = np.average(J[col], weights=J.mp)
        ok = W[col].notna()
        lg = np.average(W.loc[ok, col], weights=W.loc[ok, "mp"])
        q = Q[col].dropna()
        pct = (q < j).mean() if up else (q > j).mean()
        rows.append({"stat": col, "label": lab, "jordan": j, "league_avg": lg, "diff": j - lg,
                     "percentile": pct * 100, "jordan_txt": fmt(col, j), "league_txt": fmt(col, lg)})
    T = pd.DataFrame(rows)
    T.to_csv(config.RESULTS / "case_jordan.csv", index=False)

    per = []
    for s in seasons:
        qs = S[(S.season == s) & S.qualified].sort_values("bpm", ascending=False).reset_index(drop=True)
        r = J[J.season == s].iloc[0]
        per.append({"season": season_label(s), "team": r.team, "games": int(r.games), "mp": int(r.mp),
                    "pts_per_g": r.pts_per_g, "ts_pct": r.ts_pct, "usg_pct": r.usg_pct, "bpm": r.bpm,
                    "bpm_rank": int(qs.index[qs.name == config.CASE_PLAYER][0]) + 1, "qualified_players": len(qs)})
    Ps = pd.DataFrame(per)
    Ps.to_csv(config.RESULTS / "case_jordan_seasons.csv", index=False)
    pd.set_option("display.width", 200)
    print(f"{config.CASE_PLAYER}, {season_label(seasons[0])} to {season_label(seasons[-1])}: "
          f"{len(Q)} qualifying player-seasons in the comparison group")
    print(T[["label", "jordan_txt", "league_txt", "percentile"]].round(1).to_string(index=False))
    print(Ps.round(3).to_string(index=False))

    # Fig. 4: percentile profile
    t = T.iloc[::-1].reset_index(drop=True)
    fig, ax = plt.subplots(figsize=(COL_W, 2.7))
    ax.barh(range(len(t)), t.percentile, color=[ACCENT if s == "bpm" else MUTED for s in t.stat], height=0.66)
    for i, r in t.iterrows():
        ax.text(r.percentile + 1.2, i, f"{r.percentile:.0f}  ({r.jordan_txt})", va="center", fontsize=5.8,
                color=INK)
    ax.axvline(50, color=INK2, lw=0.7, ls="--")
    ax.set_yticks(range(len(t)), t.label, fontsize=6.5)
    ax.set_xlim(0, 125)
    ax.set_xticks([0, 25, 50, 75, 100])
    ax.set_xlabel(f"백분위 (같은 3시즌 규정 출전 선수 {len(Q)}명-시즌 대비)")
    ax.grid(axis="y", visible=False)
    save(fig, "fig4_jordan.png")


if __name__ == "__main__":
    main()
