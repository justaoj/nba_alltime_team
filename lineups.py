"""Lineup slots shared by steps 09, 12 and 13: G, G, F, F, C as on All-NBA teams (config.SLOTS).

A player can fill G if his listed positions include PG or SG, F if they include SF or PF, C if they include C.
"""
import itertools
import numpy as np
import config


def slot_groups(pos):
    """'PG-SG' -> {'G'}; 'SF-PF-C' -> {'F', 'C'}; '' -> set()"""
    return {config.SLOT_GROUP[p] for p in str(pos).split("-") if p in config.SLOT_GROUP}


def valid_fives(elig):
    """All sets of 5 distinct players that can fill G, G, F, F, C, as a sorted (M, 5) array of indices.
    elig: one set of slot groups per player."""
    G = [i for i, e in enumerate(elig) if "G" in e]
    F = [i for i, e in enumerate(elig) if "F" in e]
    C = [i for i, e in enumerate(elig) if "C" in e]
    out = set()
    for c in C:
        for f in itertools.combinations([i for i in F if i != c], 2):
            used = {c, *f}
            for g in itertools.combinations([i for i in G if i not in used], 2):
                out.add(tuple(sorted((c, *f, *g))))
    return np.array(sorted(out))


def assign_slots(pos_list):
    """pos_list: listed positions of 5 players ('PG-SG', 'C', ...).
    Returns (order, slots): player indices in display order G, G, F, F, C and the slot of each, or None.
    Within a slot, players are ordered by their first listed position (PG before SG, SF before PF)."""
    elig = [slot_groups(p) for p in pos_list]
    for perm in itertools.permutations(range(5)):
        if all(s in elig[p] for s, p in zip(config.SLOTS, perm)):
            first = {p: min((config.POSITIONS.index(x) for x in str(pos_list[p]).split("-")
                             if config.SLOT_GROUP.get(x) == s), default=9) for s, p in zip(config.SLOTS, perm)}
            slot_of = dict(zip(perm, config.SLOTS))
            rank = {"G": 0, "F": 1, "C": 2}
            order = sorted(perm, key=lambda p: (rank[slot_of[p]], first[p]))
            return order, [slot_of[p] for p in order]
    return None
