"""Step 10: Monte Carlo simulations for the candidate fives.

Every simulated game: home margin ~ Normal(h + b * (R_home - R_away) * pace/100, sigma^2), where
h, b, sigma were fitted on the target season's real games (step 8), R = net rating.
The five's R comes from the main team model (one-ball usage model, step 7); real teams use their
real net rating that season.

  1. Head-to-head vs famous dynasties (config.DYNASTIES), neutral court:
     N_SIMS single games and N_SIMS best-of-7 series per candidate and opponent.
  2. League season: the five replaces the target season's worst team, plays that team's real schedule,
     every other real game is simulated too (1,230 games), then the play-in (7-10 seeds) and a
     16-team best-of-7 bracket with home court to the better record (2-2-1-1-1). N_SIMS seasons.
     Run twice: with the five's rating fixed, and with it redrawn each season from the model's
     uncertainty (usage trade-off k bootstrap + the model's held-out error). The best five is also
     run under each of the other team models, to show how much the answer depends on the model.
Uncertainty intervals: 200 rating draws (k bootstrap x model error), 5th-95th percentile.
Outputs: results/ratings_by_model.csv, results/h2h.csv, results/league_teams.csv, results/league_five.csv,
         results/league_win_distribution.csv, results/simulation_summary.json,
         figures/fig7_head_to_head.png (paper Fig. 7), figures/fig8_season_title.png (paper Fig. 8)
Run:  python3 10_simulate.py
"""
import json
import pickle
import time
import numpy as np
import pandas as pd
from scipy.stats import norm, binom
import config
from common import load, style, save, season_label, INK, INK2, MUTED, ACCENT, COL_W, FULL_W
from teammodel import features, FIT_FEATURES, PLAYER_COLS
import matplotlib.pyplot as plt

MAIN = "usage_model"
MODEL_NAMES = {"usage_model": "원볼 사용률 모델 (주 모델)", "ridge": "릿지 회귀", "additive": "BPM 합산",
               "boosting": "그래디언트 부스팅"}


# ---------- ratings of the candidate fives ----------

def candidate_ratings(d, k, ridge, boost):
    F = features(d, "label", k=k).set_index("label")
    return pd.DataFrame({"usage_model": F.usage_model, "ridge": ridge.predict(F[FIT_FEATURES]),
                         "additive": F.bpm_sum, "boosting": boost.predict(F[FIT_FEATURES])})


def rating_draws(d, k_boot, rmse, ridge, boost, rng):
    """200 draws of each five's main-model rating: k resampled, plus the model's held-out error."""
    out = []
    z = rng.normal(size=len(k_boot))
    for kb, zb in zip(k_boot, z):
        R = candidate_ratings(d, kb, ridge, boost)[MAIN]
        out.append(R + rmse * zb)
    return pd.DataFrame(out).reset_index(drop=True)          # rows = draws, columns = labels


# ---------- game and series probabilities ----------

def p_win(diff, P, home=0.0, pace=1.0):
    """P(team wins) given rating difference (team - opponent) and home-court points (+h, 0 or -h)."""
    return norm.cdf((home + P["slope"] * diff * pace) / P["sigma"])


def p_series_neutral(p):
    return binom.sf(3, 7, p)                                  # P(at least 4 wins in 7 games)


# ---------- league simulation ----------

class League:
    def __init__(self, season, replace, P):
        T = load("team_seasons")
        G = load("games")
        t = T[T.season == season].set_index("team")
        self.teams = list(t.index)
        idx = {tm: i for i, tm in enumerate(self.teams)}
        self.replaced = t.nrtg.idxmin() if replace == "worst" else replace
        self.five = idx[self.replaced]
        self.r = t.nrtg.values.astype(float)
        self.pace = t.pace.values.astype(float)
        self.pace[self.five] = t.pace.mean()
        self.conf = t.conf.values
        g = G[G.season == season]
        self.home = g.home.map(idx).values
        self.away = g.away.map(idx).values
        self.pace_g = (self.pace[self.home] + self.pace[self.away]) / 200
        n_t = len(self.teams)
        self.H = np.zeros((len(g), n_t), np.float32)
        self.A = np.zeros((len(g), n_t), np.float32)
        self.H[np.arange(len(g)), self.home] = 1
        self.A[np.arange(len(g)), self.away] = 1
        self.P = P
        self.t = t

    def single(self, home, away, rmat, rng):
        rows = np.arange(len(home))
        pf = (self.pace[home] + self.pace[away]) / 200
        p = p_win(rmat[rows, home] - rmat[rows, away], self.P, self.P["home_court"], pf)
        won = rng.random(len(home)) < p
        return np.where(won, home, away), np.where(won, away, home)

    def series(self, a, b, key, rmat, rng):
        rows = np.arange(len(a))
        a_hi = key[rows, a] >= key[rows, b]
        hi, lo = np.where(a_hi, a, b), np.where(a_hi, b, a)
        pf = (self.pace[hi] + self.pace[lo]) / 200
        diff = rmat[rows, hi] - rmat[rows, lo]
        pattern = np.array([1, 1, -1, -1, 1, -1, 1])          # +1: higher seed at home
        p = p_win(diff[:, None], self.P, pattern[None, :] * self.P["home_court"], pf[:, None])
        hi_wins = (rng.random(p.shape) < p).sum(1)
        return np.where(hi_wins >= 4, hi, lo)

    def simulate(self, R_five, rng):
        """R_five: the five's rating per simulated season (length n). Returns per-team counters and the five's wins."""
        n, n_t = len(R_five), len(self.teams)
        rmat = np.tile(self.r, (n, 1))
        rmat[:, self.five] = R_five
        mu = self.P["home_court"] + self.P["slope"] * (rmat[:, self.home] - rmat[:, self.away]) * self.pace_g
        win = rng.random(mu.shape) < norm.cdf(mu / self.P["sigma"])
        W = win.astype(np.float32) @ self.H + (~win).astype(np.float32) @ self.A
        key = W + rng.random(W.shape) * 0.5                  # random tiebreak between equal records
        rows = np.arange(n)
        made = np.zeros((n, n_t), bool)
        top_seed = np.zeros((n, n_t), bool)
        champs = {}
        for c in ("E", "W"):
            cidx = np.where(self.conf == c)[0]
            order = cidx[np.argsort(-key[:, cidx], axis=1)]   # seeds 1..15 as team indices
            s = [order[:, i] for i in range(10)]
            top_seed[rows, s[0]] = True
            w78, l78 = self.single(s[6], s[7], rmat, rng)      # play-in: 7 hosts 8
            w910, _ = self.single(s[8], s[9], rmat, rng)       # 9 hosts 10
            w8, _ = self.single(l78, w910, rmat, rng)          # loser of 7/8 hosts winner of 9/10
            seeds = s[:6] + [w78, w8]
            for t_ in seeds:
                made[rows, t_] = True
            r1 = [self.series(seeds[0], seeds[7], key, rmat, rng), self.series(seeds[3], seeds[4], key, rmat, rng),
                  self.series(seeds[2], seeds[5], key, rmat, rng), self.series(seeds[1], seeds[6], key, rmat, rng)]
            r2 = [self.series(r1[0], r1[1], key, rmat, rng), self.series(r1[2], r1[3], key, rmat, rng)]
            champs[c] = self.series(r2[0], r2[1], key, rmat, rng)
        finals = np.zeros((n, n_t), bool)
        finals[rows, champs["E"]] = True
        finals[rows, champs["W"]] = True
        champ = np.zeros((n, n_t), bool)
        champ[rows, self.series(champs["E"], champs["W"], key, rmat, rng)] = True
        return {"wins": W.sum(0), "made": made.sum(0), "top_seed": top_seed.sum(0), "finals": finals.sum(0),
                "champ": champ.sum(0), "five_wins": W[:, self.five].astype(int)}


def run_league(lg, R_fn, n, rng):
    tot = None
    five_wins = []
    for start in range(0, n, config.SIM_CHUNK):
        m = min(config.SIM_CHUNK, n - start)
        out = lg.simulate(R_fn(m), rng)
        five_wins.append(out.pop("five_wins"))
        tot = out if tot is None else {k: tot[k] + v for k, v in out.items()}
    return {k: v / n for k, v in tot.items()}, np.concatenate(five_wins)


def main():
    style()
    t0 = time.time()
    rng = np.random.default_rng(config.SEED)
    P = json.load(open(config.RESULTS / "sim_params.json", encoding="utf-8"))
    k = json.load(open(config.RESULTS / "model_params.json", encoding="utf-8"))["k"]
    k_boot = pd.read_csv(config.RESULTS / "k_bootstrap.csv").k.values
    rmse = pd.read_csv(config.RESULTS / "team_model_cv.csv").set_index("kind").rmse["usage"]
    ridge = pickle.load(open(config.RESULTS / "ridge_model.pkl", "rb"))
    boost = pickle.load(open(config.RESULTS / "boost_model.pkl", "rb"))
    pr = pd.read_csv(config.RESULTS / "primes.csv")
    pr[["rel_3pa", "fg3_pct"]] = pr[["rel_3pa", "fg3_pct"]].fillna(0)
    C = pd.read_csv(config.RESULTS / "candidates.csv")
    desc = C.drop_duplicates("label").set_index("label").description
    d = C.merge(pr[["player_id"] + PLAYER_COLS], on="player_id", suffixes=("_c", ""))
    d["share"] = 1.0

    ratings = candidate_ratings(d, k, ridge, boost)
    ratings.to_csv(config.RESULTS / "ratings_by_model.csv")
    draws = rating_draws(d, k_boot, rmse, ridge, boost, rng)
    labels = list(ratings.index)
    best = ratings[MAIN].idxmax()
    N = config.N_SIMS
    print(f"{N:,} simulations per matchup; game model h {P['home_court']:.2f}, b {P['slope']:.3f}, sigma {P['sigma']:.2f}")
    print("\nnet rating by model:")
    print(ratings.round(1).to_string())

    # ---------- 1. head-to-head vs dynasties, neutral court ----------
    T = load("team_seasons")
    lg_pace = T[T.season == config.TARGET_SEASON].pace.mean()
    opps = []
    for s, tm in config.DYNASTIES:
        r = T[(T.season == s) & (T.team == tm)].iloc[0]
        opps.append({"season": s, "team": tm, "name": r.team_name, "record": f"{int(r.wins)}-{int(r.losses)}",
                     "nrtg": r.nrtg, "pace": r.pace})
    U_game = rng.random(N)                                   # common random numbers: same draws for every matchup
    U_series = rng.random((N, 7))
    rows = []
    for lab in labels:
        R = ratings.loc[lab, MAIN]
        for o in opps:
            pf = (o["pace"] + lg_pace) / 200
            p = float(p_win(R - o["nrtg"], P, 0.0, pf))
            pd_ = p_win(draws[lab].values - o["nrtg"], P, 0.0, pf)
            rows.append({"label": lab, "opponent": f"{season_label(o['season'])} {o['team']}", "opp_name": o["name"],
                         "opp_record": o["record"], "opp_nrtg": o["nrtg"], "rating": R,
                         "game_win_mc": float(np.mean(U_game < p)), "game_win_exact": p,
                         "game_lo": float(np.percentile(pd_, 5)), "game_hi": float(np.percentile(pd_, 95)),
                         "series_win_mc": float(np.mean((U_series < p).sum(1) >= 4)),
                         "series_win_exact": float(p_series_neutral(p)),
                         "series_lo": float(np.percentile(p_series_neutral(pd_), 5)),
                         "series_hi": float(np.percentile(p_series_neutral(pd_), 95))})
    H2H = pd.DataFrame(rows)
    H2H.to_csv(config.RESULTS / "h2h.csv", index=False)
    pd.set_option("display.width", 220)
    print(f"\nhead-to-head, neutral court, team {best}:")
    print(H2H[H2H.label == best][["opponent", "opp_record", "opp_nrtg", "game_win_mc", "game_lo", "game_hi",
                                  "series_win_mc", "series_lo", "series_hi"]].round(3).to_string(index=False))

    # ---------- 2. league season + playoffs ----------
    lg = League(config.TARGET_SEASON, config.REPLACE_TEAM, P)
    print(f"\nleague: {season_label(config.TARGET_SEASON)}, the five replaces {lg.replaced} "
          f"({lg.t.loc[lg.replaced, 'team_name']}, {int(lg.t.loc[lg.replaced, 'wins'])}-"
          f"{int(lg.t.loc[lg.replaced, 'losses'])}), conference {lg.conf[lg.five]}; {len(lg.home):,} games per season")
    team_rows, five_rows = [], []
    for lab in labels:
        for mode in ("fixed", "uncertain"):
            if mode == "fixed":
                R_fn = lambda m, R=ratings.loc[lab, MAIN]: np.full(m, R)
            else:
                R_fn = lambda m, v=draws[lab].values: rng.choice(v, m)
            res, fw = run_league(lg, R_fn, N, rng)
            five_rows.append({"label": lab, "mode": mode, "rating": ratings.loc[lab, MAIN],
                              "mean_wins": fw.mean(), "wins_p5": np.percentile(fw, 5), "wins_p95": np.percentile(fw, 95),
                              "p_73plus": np.mean(fw >= 73), "p_top_seed": res["top_seed"][lg.five],
                              "p_playoffs": res["made"][lg.five], "p_finals": res["finals"][lg.five],
                              "p_champion": res["champ"][lg.five]})
            if mode == "fixed":
                for i, tm in enumerate(lg.teams):
                    team_rows.append({"label": lab, "team": "ALL-TIME FIVE" if i == lg.five else tm,
                                      "conf": lg.conf[i], "rating": ratings.loc[lab, MAIN] if i == lg.five else lg.r[i],
                                      "mean_wins": res["wins"][i], "p_playoffs": res["made"][i],
                                      "p_top_seed": res["top_seed"][i], "p_finals": res["finals"][i],
                                      "p_champion": res["champ"][i]})
            print(f"  {lab} {mode:9s} wins {fw.mean():5.1f} [{np.percentile(fw, 5):.0f}-{np.percentile(fw, 95):.0f}]  "
                  f"P(73+) {np.mean(fw >= 73):6.1%}  P(title) {res['champ'][lg.five]:6.1%}  ({time.time() - t0:.0f} s)")
    # model sensitivity: the best five under each of the other team models (rating fixed)
    for m in [x for x in MODEL_NAMES if x != MAIN]:
        R_m = ratings.loc[best, m]
        res, fw = run_league(lg, lambda n_, R=R_m: np.full(n_, R), N, rng)
        five_rows.append({"label": best, "mode": f"model:{m}", "rating": R_m, "mean_wins": fw.mean(),
                          "wins_p5": np.percentile(fw, 5), "wins_p95": np.percentile(fw, 95),
                          "p_73plus": np.mean(fw >= 73), "p_top_seed": res["top_seed"][lg.five],
                          "p_playoffs": res["made"][lg.five], "p_finals": res["finals"][lg.five],
                          "p_champion": res["champ"][lg.five]})
        print(f"  {best} {m:10s} rating {R_m:+5.1f}  wins {fw.mean():5.1f}  P(title) {res['champ'][lg.five]:6.1%}")
    dist = {}
    for lab in labels:
        _, fw = run_league(lg, lambda n_, R=ratings.loc[lab, MAIN]: np.full(n_, R), min(N, 20_000), rng)
        dist[lab] = np.bincount(fw, minlength=83) / len(fw)
    pd.DataFrame(dist).to_csv(config.RESULTS / "league_win_distribution.csv", index_label="wins")
    LT = pd.DataFrame(team_rows)
    LF = pd.DataFrame(five_rows)
    LT.to_csv(config.RESULTS / "league_teams.csv", index=False)
    LF.to_csv(config.RESULTS / "league_five.csv", index=False)
    tb = LT[LT.label == best].sort_values("p_champion", ascending=False)
    print(f"\ntitle odds with team {best} in the league (top 6):")
    print(tb.head(6)[["team", "conf", "rating", "mean_wins", "p_finals", "p_champion"]].round(3).to_string(index=False))

    n_games = len(labels) * len(opps) * N * 2
    n_seasons = len(labels) * 2 * N + (len(MODEL_NAMES) - 1) * N + len(labels) * min(N, 20_000)
    summary = {"n_sims": N, "best": best, "replaced": lg.replaced, "replaced_name": lg.t.loc[lg.replaced, "team_name"],
               "replaced_conf": lg.conf[lg.five], "games_per_season": int(len(lg.home)),
               "h2h_simulated_games_and_series": n_games, "league_seasons_simulated": n_seasons,
               "league_games_simulated": n_seasons * int(len(lg.home)), "runtime_s": time.time() - t0}
    json.dump(summary, open(config.RESULTS / "simulation_summary.json", "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    print(f"\n{n_games:,} head-to-head games and series, {n_seasons:,} league seasons "
          f"({summary['league_games_simulated']:,} games) in {time.time() - t0:.0f} s")

    # Fig. 7: head-to-head of the best five vs the dynasties
    h = H2H[H2H.label == best].iloc[::-1].reset_index(drop=True)
    fig, ax = plt.subplots(figsize=(COL_W, 2.2))
    y = np.arange(len(h))
    ax.hlines(y + 0.14, h.game_lo * 100, h.game_hi * 100, color=MUTED, lw=2.2)
    ax.hlines(y - 0.14, h.series_lo * 100, h.series_hi * 100, color=ACCENT, lw=2.2, alpha=0.45)
    ax.scatter(h.game_win_mc * 100, y + 0.14, s=14, color=INK2, zorder=3, label="단일 경기 승률")
    ax.scatter(h.series_win_mc * 100, y - 0.14, s=16, color=ACCENT, zorder=3, label="7전 4선승 시리즈 승률")
    for i, r in h.iterrows():
        ax.text(101, i + 0.14, f"{r.game_win_mc:.0%}", va="center", fontsize=6, color=INK2)
        ax.text(101, i - 0.14, f"{r.series_win_mc:.0%}", va="center", fontsize=6, color=ACCENT)
    ax.set_yticks(y, [f"{r.opponent} ({r.opp_record}, {r.opp_nrtg:+.1f})" for r in h.itertuples()], fontsize=6.3)
    ax.set_xlim(50, 108)
    ax.set_xlabel("올타임 5인의 승률 (%), 중립 경기장, 막대 = 90% 불확실성 구간")
    ax.legend(loc="upper center", bbox_to_anchor=(0.45, -0.25), ncol=2)
    ax.grid(axis="y", visible=False)
    save(fig, "fig7_head_to_head.png")

    # Fig. 8: regular-season wins by candidate (left) and title odds of the best five under each team model (right)
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(FULL_W, 2.0), gridspec_kw={"width_ratios": [1.15, 1]})
    D = pd.read_csv(config.RESULTS / "league_win_distribution.csv", index_col="wins")
    lo_w = int(max(55, min((D[l] > 0.001).idxmax() for l in labels) - 1))
    shades = {l: (ACCENT if l == best else c) for l, c in zip(labels, [INK2, "#8a8984", MUTED, "#d8d7d2"])}
    for lab in labels:
        a1.step(D.index[lo_w:], D[lab].iloc[lo_w:] * 100, where="mid", color=shades[lab],
                lw=1.4 if lab == best else 1.0, label=f"{lab}  {desc[lab]}")
    a1.set_xlim(lo_w, 82)
    a1.set_xlabel(f"정규시즌 승수 ({season_label(config.TARGET_SEASON)} 리그, {lg.replaced} 자리, 82경기)")
    a1.set_ylabel("시즌 비율 (%)")
    a1.legend(loc="upper left", fontsize=6)
    ms = LF[(LF.label == best) & ((LF["mode"] == "fixed") | LF["mode"].str.startswith("model:"))].copy()
    ms["model"] = ms["mode"].str.replace("model:", "").replace({"fixed": MAIN})
    ms = ms.set_index("model").loc[list(MODEL_NAMES)[::-1]]
    a2.barh(range(len(ms)), ms.p_champion * 100, color=[ACCENT if m == MAIN else MUTED for m in ms.index], height=0.6)
    for i, (m, r) in enumerate(ms.iterrows()):
        a2.text(r.p_champion * 100 + 1.5, i, f"{r.p_champion:.1%}", va="center", fontsize=6)
    a2.set_yticks(range(len(ms)), [f"{MODEL_NAMES[m]} ({r.rating:+.1f})" for m, r in ms.iterrows()], fontsize=6.5)
    a2.set_xlim(0, 115)
    a2.set_xlabel(f"팀 {best}의 우승 확률 (%), 팀 모델별")
    a2.grid(axis="y", visible=False)
    fig.tight_layout()
    save(fig, "fig8_season_title.png")


if __name__ == "__main__":
    main()
