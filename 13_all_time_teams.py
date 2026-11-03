"""Step 13: All-Time First, Second and Third Teams from the 3-season primes (paper sections IV-VII).

Slots: G, G, F, F, C as on real All-NBA teams (lineups.py, same rule as step 09). A player is a G if his
prime lists PG or SG, an F if it lists SF or PF, a C if it lists C.
Pool: the top POOL primes by BPM (60, as in step 09). Every valid G-G-F-F-C five is scored by the
one-ball usage model (the paper's main team model).
  1st team = the highest-rated five; its players leave the pool.
  2nd team = the highest-rated five of the rest; its players leave the pool.
  3rd team = the highest-rated five of the rest.
  pick     = config.EXPECTED_FIVE, the author's pick before the analysis, for comparison.
Simulations (N_SIMS each, game model from step 08):
  - the teams against each other, neutral court: single games and best-of-7 series
  - each team against the dynasties in config.DYNASTIES, neutral court
  - each team in the target-season league in the worst team's place: regular season, play-in, playoffs
    (rating fixed, then with model uncertainty)
Outputs: results/all_time_teams.csv, results/all_time_h2h.csv, results/all_time_league.csv,
         results/all_time_win_distribution.csv, results/all_time_summary.json,
         figures/fig10_all_time_teams.png, figures/fig11_all_time_season.png, figures/fig12_all_time_dynasties.png,
         figures/fig13_model_sensitivity.png, paper/figures/ (fig11, fig12, fig13), paper/tables/table4_all_time.tex, paper/tables/all_time_cards.tex,
         paper/numbers_teams.tex
Needs steps 02-10 first (run_all.py).
Run:  python3 13_all_time_teams.py
"""
import _venv
_venv.ensure()                      # re-runs this script inside .venv if needed
import importlib
import json
import pickle
import shutil
import time
import numpy as np
import pandas as pd
import config
from common import style, save, season_label, INK, INK2, MUTED, ACCENT, FULL_W, COL_W
from lineups import slot_groups, valid_fives, assign_slots
from teammodel import PLAYER_COLS
import matplotlib.pyplot as plt

search = importlib.import_module("09_search")      # score(), POOL
sim = importlib.import_module("10_simulate")       # ratings, game model, League

MAIN = sim.MAIN
TEAMS = ["1st", "2nd", "3rd"]
ALL = TEAMS + ["pick"]
TEAM_KO = {"1st": "올타임 퍼스트 팀", "2nd": "올타임 세컨드 팀", "3rd": "올타임 서드 팀", "pick": "분석 전 예상 5인"}
TEAM_EN = {"1st": "FIRST TEAM", "2nd": "SECOND TEAM", "3rd": "THIRD TEAM", "pick": "MY PICK"}
MACRO = {"1st": "First", "2nd": "Second", "3rd": "Third", "pick": "Pick"}
COLOR = {"1st": ACCENT, "2nd": INK2, "3rd": "#a3a29c", "pick": "#c8693c"}
MARKER = {"1st": "o", "2nd": "s", "3rd": "^", "pick": "D"}


def years(first, last):
    """Seasons named by the year they end -> '1987--90' (prime from 1987-88 to 1989-90)."""
    return f"{int(first) - 1}--{str(int(last))[-2:]}"


def tex(s):
    return str(s).replace("&", "\\&").replace("%", "\\%").replace("_", "\\_")


def plab(p):
    """Percent label for figures; never rounds a probability below 1 up to 100%."""
    s = f"{p:.1%}"
    return ">99.9%" if s == "100.0%" and p < 1 else s


def pct(p, nd=1):
    s = f"{100 * p:.{nd}f}"
    if s == f"{100:.{nd}f}" and p < 1:
        return f">{100 - 10 ** -nd:.{nd}f}\\%"
    return s + "\\%"


def main():
    style()
    t0 = time.time()
    rng = np.random.default_rng(config.SEED)
    N = config.N_SIMS
    P = json.load(open(config.RESULTS / "sim_params.json", encoding="utf-8"))
    k = json.load(open(config.RESULTS / "model_params.json", encoding="utf-8"))["k"]
    k_boot = pd.read_csv(config.RESULTS / "k_bootstrap.csv").k.values
    rmse = pd.read_csv(config.RESULTS / "team_model_cv.csv").set_index("kind").rmse["usage"]
    ridge = pickle.load(open(config.RESULTS / "ridge_model.pkl", "rb"))
    boost = pickle.load(open(config.RESULTS / "boost_model.pkl", "rb"))
    pr = pd.read_csv(config.RESULTS / "primes.csv")
    pr[["rel_3pa", "fg3_pct"]] = pr[["rel_3pa", "fg3_pct"]].fillna(0)
    pr = pr.fillna(pr.median(numeric_only=True))
    pd.set_option("display.width", 220)

    # ---------- search: 1st, 2nd, 3rd team ----------
    pool = pr.head(search.POOL).reset_index(drop=True)
    combos = valid_fives([slot_groups(p) for p in pool.prime_pos])
    print(f"pool of {len(pool)} primes -> {len(combos):,} valid G-G-F-F-C fives")
    F = search.score(pool, combos, ridge, boost, k)
    taken = []
    rows = []
    for team in TEAMS:
        ok = ~np.isin(combos, taken).any(axis=1)
        j = F.usage_model.where(ok).idxmax()
        idx = combos[j]
        order, slots = assign_slots(list(pool.prime_pos.iloc[idx]))
        for slot, p in zip(slots, order):
            r = pool.iloc[idx[p]]
            rows.append({"team": team, "slot": slot, "player_id": r.player_id, "name": r["name"], "prime": r.prime,
                         "prime_pos": r.prime_pos, "bpm_rank": int(r["rank"]), **{c: r[c] for c in PLAYER_COLS}})
        taken += list(idx)
        print(f"{team}: {F.usage_model[j]:+.1f} (BPM sum {F.bpm_sum[j]:+.1f}, from {int(ok.sum()):,} fives)  "
              + " | ".join(pool.name[idx[p]] for p in order))
    pick = pr[pr.name.isin(config.EXPECTED_FIVE)]
    missing = set(config.EXPECTED_FIVE) - set(pick.name)
    if missing:
        raise SystemExit(f"EXPECTED_FIVE: no prime found for {missing}")
    order, slots = assign_slots(list(pick.prime_pos))
    for slot, p in zip(slots, order):
        r = pick.iloc[p]
        rows.append({"team": "pick", "slot": slot, "player_id": r.player_id, "name": r["name"], "prime": r.prime,
                     "prime_pos": r.prime_pos, "bpm_rank": int(r["rank"]), **{c: r[c] for c in PLAYER_COLS}})
    T = pd.DataFrame(rows)
    T = T.merge(pr[["player_id", "first", "last"]], on="player_id", how="left")
    T.to_csv(config.RESULTS / "all_time_teams.csv", index=False)
    print("\n" + T[["team", "slot", "name", "prime", "prime_pos", "bpm_rank", "bpm", "usg_pct"]]
          .round(1).to_string(index=False))

    # ---------- ratings ----------
    d = T.rename(columns={"team": "label"}).copy()
    d["share"] = 1.0
    ratings = sim.candidate_ratings(d, k, ridge, boost).loc[ALL]
    draws = sim.rating_draws(d, k_boot, rmse, ridge, boost, rng)
    print("\nnet rating by model:")
    print(ratings.round(1).to_string())

    # ---------- head-to-head: the teams against each other and against the dynasties ----------
    TS = sim.load("team_seasons")
    lg_pace = TS[TS.season == config.TARGET_SEASON].pace.mean()
    opps = [{"name": TEAM_KO[t], "label": t, "nrtg": None, "pace": lg_pace} for t in ALL]
    for s, tm in config.DYNASTIES:
        r = TS[(TS.season == s) & (TS.team == tm)].iloc[0]
        opps.append({"name": f"{season_label(s)} {tm} ({int(r.wins)}-{int(r.losses)})", "label": None,
                     "nrtg": r.nrtg, "pace": r.pace})
    U_game = rng.random(N)                                   # common random numbers for every matchup
    U_series = rng.random((N, 7))
    H = []
    for i, team in enumerate(ALL):
        R = ratings.loc[team, MAIN]
        for o in opps:
            if o["label"] is not None and ALL.index(o["label"]) <= i:
                continue                                     # each pair of teams once, the better team first
            o_r = ratings.loc[o["label"], MAIN] if o["label"] else o["nrtg"]
            o_d = draws[o["label"]].values if o["label"] else o["nrtg"]
            pf = (o["pace"] + lg_pace) / 200
            p = float(sim.p_win(R - o_r, P, 0.0, pf))
            pd_ = sim.p_win(draws[team].values - o_d, P, 0.0, pf)
            H.append({"team": team, "opponent": o["name"], "opp_label": o["label"] or "", "opp_rating": o_r, "rating": R,
                      "game_win_mc": float(np.mean(U_game < p)), "game_win_exact": p,
                      "game_lo": float(np.percentile(pd_, 5)), "game_hi": float(np.percentile(pd_, 95)),
                      "series_win_mc": float(np.mean((U_series < p).sum(1) >= 4)),
                      "series_win_exact": float(sim.p_series_neutral(p))})
    H = pd.DataFrame(H)
    H.to_csv(config.RESULTS / "all_time_h2h.csv", index=False)
    print("\nhead-to-head, neutral court (single game, best-of-7):")
    print(H[["team", "opponent", "rating", "opp_rating", "game_win_mc", "series_win_mc"]].round(3).to_string(index=False))

    # ---------- league season + playoffs ----------
    lg = sim.League(config.TARGET_SEASON, config.REPLACE_TEAM, P)
    print(f"\nleague: {season_label(config.TARGET_SEASON)}, each team replaces {lg.replaced}; {N:,} seasons per run")
    L = []
    dist = {}
    for team in ALL:
        R = ratings.loc[team, MAIN]
        for mode in ("fixed", "uncertain"):
            if mode == "fixed":
                R_fn = lambda m, R=R: np.full(m, R)
            else:
                R_fn = lambda m, v=draws[team].values: rng.choice(v, m)
            res, fw = sim.run_league(lg, R_fn, N, rng)
            if mode == "fixed":
                dist[team] = np.bincount(fw, minlength=83) / len(fw)
            L.append({"team": team, "mode": mode, "rating": R, "mean_wins": fw.mean(),
                      "wins_p5": np.percentile(fw, 5), "wins_p95": np.percentile(fw, 95),
                      "p_82_0": np.mean(fw == 82), "p_playoffs": res["made"][lg.five],
                      "p_finals": res["finals"][lg.five], "p_champion": res["champ"][lg.five]})
            print(f"  {team} {mode:9s} rating {R:+5.1f}  wins {fw.mean():5.1f} [{np.percentile(fw, 5):.0f}-"
                  f"{np.percentile(fw, 95):.0f}]  P(82-0) {np.mean(fw == 82):6.2%}  "
                  f"P(title) {res['champ'][lg.five]:7.3%}  ({time.time() - t0:.0f} s)")
    LF = pd.DataFrame(L)
    LF.to_csv(config.RESULTS / "all_time_league.csv", index=False)
    D = pd.DataFrame(dist)
    D.to_csv(config.RESULTS / "all_time_win_distribution.csv", index_label="wins")

    n_games = len(H) * N * 2
    summary = {"pool": int(len(pool)), "n_fives": int(len(combos)), "n_sims": N,
               "teams": {t: list(T[T.team == t].name) for t in ALL},
               "ratings": ratings.round(3).to_dict(orient="index"),
               "h2h_simulated_games_and_series": n_games, "league_seasons_simulated": len(L) * N,
               "runtime_s": time.time() - t0}
    json.dump(summary, open(config.RESULTS / "all_time_summary.json", "w", encoding="utf-8"),
              ensure_ascii=False, indent=1)
    print(f"\n{n_games:,} head-to-head games and series, {len(L) * N:,} league seasons in {time.time() - t0:.0f} s")

    # ---------- Fig. 10: the three teams ----------
    fig, axes = plt.subplots(1, 3, figsize=(FULL_W, 2.45), sharex=True)
    xmax = T.bpm.max() * 1.2
    lf = LF[LF["mode"] == "fixed"].set_index("team")
    for ax, team in zip(axes, TEAMS):
        t = T[T.team == team].iloc[::-1].reset_index(drop=True)      # G on top
        y = np.arange(len(t))
        ax.barh(y, t.bpm, color=ACCENT, height=0.6)
        for i, r in t.iterrows():
            ax.text(r.bpm + 0.2, i, f"{r.bpm:+.1f}", va="center", fontsize=6, color=INK)
        ax.set_yticks(y, [f"{r.slot}  {r['name']}\n{r.prime.replace(' to ', '~')}" for _, r in t.iterrows()],
                      fontsize=5.8)
        ax.set_xlim(0, xmax)
        r = lf.loc[team]
        ax.set_title(f"{TEAM_KO[team]} ({ratings.loc[team, MAIN]:+.1f})\n"
                     f"평균 {r.mean_wins:.1f}승 · 우승 {plab(r.p_champion)}", fontsize=6.8, color=INK, loc="left")
        ax.grid(axis="y", visible=False)
    fig.supxlabel(f"전성기 BPM (3년 기준, 100포세션당, 평균 선수 대비) · 괄호 = 팀 레이팅 · 승수와 우승 확률 = "
                  f"{season_label(config.TARGET_SEASON)} 리그 {lg.replaced} 자리, {N:,}시즌", fontsize=6.8, color=INK2)
    fig.tight_layout(w_pad=0.8)
    save(fig, "fig10_all_time_teams.png")

    # ---------- Fig. 11: regular-season wins and title odds in the target season (one column) ----------
    lu = LF[LF["mode"] == "uncertain"].set_index("team")
    fig, (a1, a2) = plt.subplots(2, 1, figsize=(COL_W, 3.5), gridspec_kw={"height_ratios": [1.5, 1]})
    lo_w = int(max(55, min((D[t] > 0.001).idxmax() for t in ALL) - 1))
    for team in ALL:
        a1.step(D.index[lo_w:], D[team].iloc[lo_w:] * 100, where="mid", color=COLOR[team],
                lw=1.5 if team == "1st" else 1.1, ls="--" if team == "pick" else "-",
                label=f"{TEAM_KO[team]} ({ratings.loc[team, MAIN]:+.1f})")
    a1.set_xlim(lo_w, 82)
    a1.set_xlabel(f"정규시즌 승수 ({season_label(config.TARGET_SEASON)} 리그, {lg.replaced} 자리, 82경기)")
    a1.set_ylabel("시즌 비율 (%)")
    a1.legend(loc="upper left", fontsize=6)
    y = np.arange(len(ALL))[::-1]
    for yy, team in zip(y, ALL):
        a2.barh(yy, lf.loc[team, "p_champion"] * 100, color=COLOR[team], height=0.6)
        a2.text(lf.loc[team, "p_champion"] * 100 + 1.5, yy,
                f"{plab(lf.loc[team, 'p_champion'])} ({plab(lu.loc[team, 'p_champion'])})",
                va="center", fontsize=6, color=INK)
    a2.set_yticks(y, [TEAM_KO[t] for t in ALL], fontsize=6.3)
    a2.set_xlim(0, 135)
    a2.set_xticks([0, 25, 50, 75, 100])
    a2.set_xlabel(f"우승 확률 (%), {N:,}시즌 · 괄호 = 모델 불확실성 포함")
    a2.grid(axis="y", visible=False)
    fig.tight_layout(h_pad=1.0)
    save(fig, "fig11_all_time_season.png")

    # ---------- Fig. 13: First Team title odds under each team model (from step 10) ----------
    R10 = pd.read_csv(config.RESULTS / "ratings_by_model.csv", index_col=0)
    L10 = pd.read_csv(config.RESULTS / "league_five.csv")
    b10 = R10[MAIN].idxmax()
    ms = L10[(L10.label == b10) & ((L10["mode"] == "fixed") | L10["mode"].str.startswith("model:"))].copy()
    ms["model"] = ms["mode"].str.replace("model:", "").replace({"fixed": MAIN})
    ms = ms.set_index("model").loc[list(sim.MODEL_NAMES)[::-1]]
    fig, ax = plt.subplots(figsize=(COL_W, 1.55))
    ax.barh(range(len(ms)), ms.p_champion * 100, color=[ACCENT if m == MAIN else MUTED for m in ms.index], height=0.6)
    for i_, (m_, r_) in enumerate(ms.iterrows()):
        ax.text(r_.p_champion * 100 + 1.5, i_, f"{plab(r_.p_champion)} · 평균 {r_.mean_wins:.1f}승", va="center",
                fontsize=6, color=INK)
    ax.set_yticks(range(len(ms)), [f"{sim.MODEL_NAMES[m_]} ({r_.rating:+.1f})" for m_, r_ in ms.iterrows()], fontsize=6.2)
    ax.set_xlim(0, 135)
    ax.set_xticks([0, 25, 50, 75, 100])
    ax.set_xlabel("퍼스트 팀의 우승 확률 (%), 팀 모델별 (괄호 = 예측 넷레이팅)")
    ax.grid(axis="y", visible=False)
    save(fig, "fig13_model_sensitivity.png")

    # ---------- Fig. 12: single-game win probability against the dynasties ----------
    Hd = H[H.opp_label == ""]
    dyn = list(dict.fromkeys(Hd.opponent))
    fig, ax = plt.subplots(figsize=(COL_W, 2.5))
    yb = np.arange(len(dyn))[::-1]
    offs = {t: o for t, o in zip(ALL, [0.27, 0.09, -0.09, -0.27])}
    for team in ALL:
        h = Hd[Hd.team == team].set_index("opponent").loc[dyn]
        kw = dict(color=COLOR[team]) if team != "pick" else dict(facecolor="white", edgecolor=COLOR[team], lw=1)
        ax.scatter(h.game_win_mc * 100, yb + offs[team], s=14, marker=MARKER[team], zorder=3, **kw,
                   label=f"{TEAM_KO[team]} ({ratings.loc[team, MAIN]:+.1f})")
    ax.set_yticks(yb, dyn, fontsize=6.2)
    ax.set_xlim(min(70, np.floor(Hd.game_win_mc.min() * 20) * 5), 100)
    ax.set_xlabel("단일 경기 승률 (%), 중립 경기장")
    ax.legend(loc="upper center", bbox_to_anchor=(0.42, -0.2), ncol=2, fontsize=6)
    ax.grid(axis="y", visible=False)
    save(fig, "fig12_all_time_dynasties.png")

    # ---------- paper assets ----------
    for f in ("fig11_all_time_season.png", "fig12_all_time_dynasties.png", "fig13_model_sensitivity.png"):
        shutil.copy(config.FIGURES / f, config.PAPER / "figures" / f)
    bulls = [o for o in dyn if o.startswith("1995-96")]
    # Table IV: the four fives
    lines = []
    for team in ALL:
        t = T[T.team == team]
        names = ", ".join(tex(n.split(" ", 1)[-1]) for n in t.name)
        f, u = lf.loc[team], lu.loc[team]
        h = Hd[Hd.team == team]
        bs = h[h.opponent.isin(bulls)].series_win_mc.iloc[0]
        lines.append(f"{TEAM_KO[team].replace('올타임 ', '')} & {names} & {ratings.loc[team, MAIN]:+.1f} & {f.mean_wins:.1f} "
                     f"[{f.wins_p5:.0f}--{f.wins_p95:.0f}] & {pct(f.p_82_0)} & {pct(f.p_champion)} ({pct(u.p_champion)}) "
                     f"& {pct(h.game_win_mc.mean())} & {pct(bs)} \\\\")
    (config.PAPER / "tables" / "table4_all_time.tex").write_text(
        "\\begin{table*}[t]\n\\centering\n\\caption{올타임 팀과 분석 전 예상 5인의 시뮬레이션 결과 (각 "
        f"{N:,}회)}}\n\\label{{tab:alltime}}\n\\footnotesize\n\\setlength{{\\tabcolsep}}{{3pt}}\n"
        "\\begin{tabular}{@{}llrrrrrr@{}}\n\\toprule\n"
        "팀 & 선수 (G, G, F, F, C) & 넷레이팅 & \\shortstack{정규시즌 승수\\\\{[}90\\% 구간{]}} & 82-0 & "
        "\\shortstack{우승 확률\\\\(불확실성 포함)} & \\shortstack{왕조 상대\\\\경기 승률} & "
        "\\shortstack{1995--96 불스\\\\상대 시리즈} \\\\\n\\midrule\n" + "\n".join(lines) +
        "\n\\bottomrule\n\\end{tabular}\n\n\\vspace{2pt}\n\\parbox{0.95\\textwidth}{\\scriptsize "
        f"넷레이팅은 원볼 사용률 모델 예측. 정규시즌·우승 확률은 {season_label(config.TARGET_SEASON)} 리그에서 "
        f"{tex(lg.t.loc[lg.replaced, 'team_name'])} 자리를 대신한 결과이며, 괄호 안은 모델 불확실성을 포함한 값. "
        "왕조 상대 경기 승률은 6개 왕조 상대 단일 경기 승률의 평균 (중립 경기장).}\n\\end{table*}\n", encoding="utf-8")
    # All-NBA style cards (photos: paper/figures/players/<player_id>.jpg or .png)
    rows_tex = []
    for team in TEAMS:
        t = T[T.team == team]
        rows_tex.append(f"\\teamheader{{{TEAM_EN[team]} \\quad ({TEAM_KO[team].replace('올타임 ', '')}, "
                        f"팀 레이팅 {ratings.loc[team, MAIN]:+.1f})}}\\\\[3pt]")
        cards = [f"\\playercard{{{r.player_id}}}{{{tex(r['name'].upper())}}}{{{r.slot}}}{{{years(r['first'], r['last'])}}}"
                 f"{{{r.bpm:+.1f}}}" for _, r in t.iterrows()]
        rows_tex.append("\\hfill".join(cards) + "\\\\[8pt]")
    (config.PAPER / "tables" / "all_time_cards.tex").write_text(
        "\\begin{figure*}[t]\n\\centering\n" + "\n".join(rows_tex) +
        "\n\\caption{올타임 퍼스트·세컨드·서드 팀 (G-G-F-F-C). 카드: 포지션 · 전성기 (3년 기준) · BPM.}\n"
        "\\label{fig:cards}\n\\end{figure*}\n", encoding="utf-8")
    # numbers_teams.tex
    nums = {}
    for team in ALL:
        m, f, u = MACRO[team], lf.loc[team], lu.loc[team]
        h = Hd[Hd.team == team]
        nums.update({f"{m}TeamNames": ", ".join(tex(n) for n in T[T.team == team].name),
                     f"{m}TeamRating": f"{ratings.loc[team, MAIN]:+.1f}",
                     f"{m}TeamWins": f"{f.mean_wins:.1f}",
                     f"{m}TeamWinsRange": f"{f.wins_p5:.0f}--{f.wins_p95:.0f}",
                     f"{m}TeamPerfect": pct(f.p_82_0),
                     f"{m}TeamTitle": pct(f.p_champion),
                     f"{m}TeamTitleUnc": pct(u.p_champion),
                     f"{m}TeamDynGame": pct(h.game_win_mc.mean(), 0),
                     f"{m}TeamDynGameMin": pct(h.game_win_mc.min(), 0),
                     f"{m}TeamDynGameMax": pct(h.game_win_mc.max(), 0),
                     f"{m}TeamBullsSeries": pct(h[h.opponent.isin(bulls)].series_win_mc.iloc[0])})
    for a_, b_ in [("1st", "2nd"), ("1st", "3rd"), ("2nd", "3rd"), ("1st", "pick")]:
        h = H[(H.team == a_) & (H.opp_label == b_)].iloc[0]
        nums[f"{MACRO[a_]}Vs{MACRO[b_]}Game"] = pct(h.game_win_mc, 0)
        nums[f"{MACRO[a_]}Vs{MACRO[b_]}Series"] = pct(h.series_win_mc, 0)
    nums["NAllTimeSeasons"] = f"{len(L) * N:,}".replace(",", "{,}")
    nums["NAllTimeGames"] = f"{n_games:,}".replace(",", "{,}")
    (config.PAPER / "numbers_teams.tex").write_text(
        "% generated by 13_all_time_teams.py - do not edit; re-run the script instead\n" +
        "\n".join(f"\\newcommand{{\\{k_}}}{{{v}}}" for k_, v in nums.items()) + "\n", encoding="utf-8")
    print(f"wrote paper/tables/table4_all_time.tex, paper/tables/all_time_cards.tex, "
          f"paper/numbers_teams.tex ({len(nums)} numbers), paper/figures (fig11, fig12, fig13)")


if __name__ == "__main__":
    main()
