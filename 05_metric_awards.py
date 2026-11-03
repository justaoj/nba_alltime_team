"""Step 5: does the metric agree with expert judgment? (MVP voting and All-NBA First Team)

For every season 1979-80 to 2025-26, rank the qualifying players by each metric and check:
  - how often the MVP ranks in the metric's top 3 that season
  - how often an All-NBA First Team player ranks in the metric's top 10
Voters also look at box scores, so agreement is not proof of accuracy; it shows the metric
picks the players experts saw as the best, while Step 4 shows it predicts team results.
Outputs: results/metric_awards.csv, results/mvp_ranks.csv, figures/fig3_awards.png (paper Fig. 3)
Run:  python3 05_metric_awards.py
"""
import numpy as np
import pandas as pd
import config
from common import style, save, INK, INK2, MUTED, ACCENT, COL_W
import matplotlib.pyplot as plt

METRICS = {"BPM": "bpm", "WS/48": "ws_per_48", "PER": "per",
           "박스스코어 합": "box5", "득점 (PTS/G)": "pts_per_g"}


def main():
    style()
    S = pd.read_csv(config.RESULTS / "player_season_one.csv")
    S["box5"] = S[["pts_per_g", "trb_per_g", "ast_per_g", "stl_per_g", "blk_per_g"]].sum(axis=1)
    S["awards"] = S["awards"].fillna("")
    Q = S[S.qualified].copy()
    for lab, col in METRICS.items():
        Q[f"rank_{col}"] = Q.groupby("season")[col].rank(ascending=False, method="min")
    mvp = Q[Q.awards.str.contains(r"(?:^|,)MVP-1(?:,|$)")]
    nba1 = Q[Q.awards.str.contains(r"(?:^|,)NBA1(?:,|$)")]
    print(f"{len(mvp)} MVPs and {len(nba1)} All-NBA First Team selections among qualifying players, "
          f"{Q.season.min()}-{Q.season.max()}")

    rows = []
    for lab, col in METRICS.items():
        rows.append({"metric": lab, "column": col,
                     "mvp_top1": (mvp[f"rank_{col}"] <= 1).mean(),
                     "mvp_top3": (mvp[f"rank_{col}"] <= 3).mean(),
                     "mvp_median_rank": mvp[f"rank_{col}"].median(),
                     "nba1_top10": (nba1[f"rank_{col}"] <= 10).mean()})
    R = pd.DataFrame(rows)
    R.to_csv(config.RESULTS / "metric_awards.csv", index=False)
    mvp[["season", "name"] + [f"rank_{c}" for c in METRICS.values()]].to_csv(config.RESULTS / "mvp_ranks.csv", index=False)
    print(R.round(3).to_string(index=False))

    # Fig. 3: grouped bars, two agreement rates per metric
    t = R.iloc[::-1].reset_index(drop=True)
    y = np.arange(len(t))
    h = 0.36
    fig, ax = plt.subplots(figsize=(COL_W, 2.0))
    ax.barh(y + h / 2, t.mvp_top3 * 100, height=h, color=ACCENT, label="MVP가 지표 상위 3위 안")
    ax.barh(y - h / 2, t.nba1_top10 * 100, height=h, color=MUTED, label="All-NBA 1st팀이 지표 상위 10위 안")
    for i, r in t.iterrows():
        ax.text(r.mvp_top3 * 100 + 1, i + h / 2, f"{r.mvp_top3:.0%}", va="center", fontsize=6, color=INK)
        ax.text(r.nba1_top10 * 100 + 1, i - h / 2, f"{r.nba1_top10:.0%}", va="center", fontsize=6, color=INK2)
    ax.set_yticks(y, t.metric, fontsize=6.8)
    ax.set_xlim(0, 108)
    ax.set_xlabel(f"일치율 (%), {Q.season.min() - 1}-{str(Q.season.min())[2:]}~"
                  f"{Q.season.max() - 1}-{str(Q.season.max())[2:]} 시즌")
    ax.legend(loc="upper center", bbox_to_anchor=(0.42, -0.3), ncol=1)
    ax.grid(axis="y", visible=False)
    save(fig, "fig3_awards.png")


if __name__ == "__main__":
    main()
