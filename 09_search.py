"""Step 9: search every position-valid five and pick the candidate teams.

Pool: the top POOL primes by BPM. A five is valid if it fills the All-NBA slots G, G, F, F, C
(config.SLOTS): a player can be a G if his prime lists PG or SG, an F if it lists SF or PF, a C if it lists C.
No player appears twice.
Every valid five is scored by each model (all players at 48 minutes, share = 1).
Candidates:
  A  best players      highest sum of BPM
  B  best fit          highest one-ball usage model rating (the main model, see step 10)
  R  best by ridge     highest ridge prediction; dropped when it is the same five as A or B
  C  box-score pick    highest sum of PTS+REB+AST+STL+BLK per game (how fans and fantasy games rank)
  D  fan five          config.FAN_FIVE
Also reports the price of positions: the best five with no position rule (top 40 primes).
Outputs: results/candidates.csv, results/search_top.csv
Run:  python3 09_search.py
"""
import itertools
import json
import pickle
import numpy as np
import pandas as pd
import config
from lineups import slot_groups, valid_fives, assign_slots
from teammodel import FIT_FEATURES

POOL = 60


def score(pr, combos, ridge, boost, k):
    """combos: (M, 5) array of row indices into pr. Returns a DataFrame of features and model scores."""
    g = lambda c: pr[c].to_numpy(dtype=float)[combos]
    u = g("usg_pct")
    out = {
        "bpm_sum": g("bpm").sum(1), "obpm_sum": g("obpm").sum(1), "dbpm_sum": g("dbpm").sum(1),
        "usg_load": u.sum(1), "drb_load": g("drb_pct").sum(1), "ast_load": g("ast_pct").sum(1),
        "spacing": (g("rel_3pa") * 100).sum(1), "rim": g("blk_pct").sum(1),
        "box5": (g("pts_per_g") + g("trb_per_g") + g("ast_per_g") + g("stl_per_g") + g("blk_per_g")).sum(1),
    }
    u_new = u * 100 / u.sum(1, keepdims=True)
    ortg_new = g("rel_ortg") - k * (u_new - u)
    out["usage_model"] = (u_new * ortg_new).sum(1) / 100 + out["dbpm_sum"]
    F = pd.DataFrame(out)
    F["ridge"] = ridge.predict(F[FIT_FEATURES])
    F["boosting"] = boost.predict(F[FIT_FEATURES])
    return F


def main():
    pr = pd.read_csv(config.RESULTS / "primes.csv")
    pr[["rel_3pa", "fg3_pct"]] = pr[["rel_3pa", "fg3_pct"]].fillna(0)
    pr = pr.fillna(pr.median(numeric_only=True))
    params = json.load(open(config.RESULTS / "model_params.json", encoding="utf-8"))
    ridge = pickle.load(open(config.RESULTS / "ridge_model.pkl", "rb"))
    boost = pickle.load(open(config.RESULTS / "boost_model.pkl", "rb"))
    k = params["k"]

    pool = pr.head(POOL).reset_index(drop=True)
    combos = valid_fives([slot_groups(p) for p in pool.prime_pos.fillna("")])
    print(f"pool of {POOL} primes -> {len(combos):,} valid G-G-F-F-C fives")

    F = score(pool, combos, ridge, boost, k)
    names = pool.name.to_numpy()
    F["players"] = [" | ".join(names[c]) for c in combos]
    objectives = {"A": ("bpm_sum", "최고 선수 (BPM 합)"), "B": ("usage_model", "최적 조합 (원볼 모델)"),
                  "R": ("ridge", "릿지 최적"), "C": ("box5", "박스스코어 합")}
    top = []
    rows = []
    chosen = {}
    for lab, (col, desc) in objectives.items():
        best = F[col].idxmax()
        top.append(F.sort_values(col, ascending=False).head(20).assign(objective=lab))
        idx = combos[best]
        same = [l for l, b in chosen.items() if b == best]
        if same:
            print(f"\n{lab} {desc}: same five as {same[0]}, not added again")
            continue
        chosen[lab] = best
        rows.append((lab, desc, idx))
        print(f"\n{lab} {desc}: {F.players[best]}")
        print("   " + ", ".join(f"{c} {F.loc[best, c]:+.1f}" for c in ["bpm_sum", "usage_model", "ridge", "boosting", "box5", "usg_load"]))
    pd.concat(top).to_csv(config.RESULTS / "search_top.csv", index=False)

    # price of positions: best five with no position rule, from the top 40 primes (658,008 fives)
    free = np.array(list(itertools.combinations(range(40), 5)))
    Ff = score(pool, free, ridge, boost, k)
    j = Ff.usage_model.idxmax()
    print(f"\nno position rule (top 40): {' | '.join(names[free[j]])}, one-ball {Ff.usage_model[j]:+.1f} "
          f"-> positions cost {Ff.usage_model[j] - F.usage_model.max():.1f} points")
    json.dump({"pool": POOL, "n_fives": int(len(combos)), "free_n_fives": int(len(free)),
               "free_best": list(names[free[j]]), "free_best_rating": float(Ff.usage_model[j]),
               "best_rating": float(F.usage_model.max()),
               "price_of_positions": float(Ff.usage_model[j] - F.usage_model.max())},
              open(config.RESULTS / "search_summary.json", "w", encoding="utf-8"), ensure_ascii=False, indent=1)

    fan = pr[pr.name.isin(config.FAN_FIVE)]
    missing = set(config.FAN_FIVE) - set(fan.name)
    if missing:
        print("fan five: no prime found for", missing)
    if len(fan) == 5:
        rows.append(("D", "팬 선정 올타임 5인", None))

    cand = []
    for lab, desc, idx in rows:
        players = pool.iloc[idx] if idx is not None else fan
        fit = assign_slots(players.prime_pos.fillna("").tolist())
        order, slots = fit if fit else (list(range(5)), ["-"] * 5)
        for k_, (slot, i) in enumerate(zip(slots, order)):
            p = players.iloc[i]
            cand.append({"label": lab, "description": desc, "order": k_, "slot": slot, "player_id": p.player_id,
                         "name": p["name"], "prime": p.prime, "bpm": p.bpm, "obpm": p.obpm, "dbpm": p.dbpm,
                         "usg_pct": p.usg_pct, "rel_ortg": p.rel_ortg})
    C = pd.DataFrame(cand).sort_values(["label", "order"])
    C.to_csv(config.RESULTS / "candidates.csv", index=False)
    print("\n" + C[["label", "slot", "name", "prime", "bpm", "usg_pct"]].round(1).to_string(index=False))


if __name__ == "__main__":
    main()
