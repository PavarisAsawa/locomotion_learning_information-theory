"""Pairwise tests, multiplicity correction and Compact Letter Display.

Shared by the significance figures so the pooled (n=150) and per-policy (n=30)
analyses run through identical machinery and only the test differs.
"""

from itertools import combinations

import numpy as np
from scipy import stats


def welch_z_pvalue(a, b):
    """Two-sided Welch z-test. Appropriate once n is large (n=150 here).

    Uses norm.sf(|z|) rather than 1 - norm.cdf(|z|): cdf(33) rounds to 1.0 in
    float64, so the subtraction underflows to exactly 0 for the strongest
    pairs. sf computes the upper tail directly and stays accurate to ~1e-238.
    """
    a, b = np.asarray(a, float), np.asarray(b, float)
    se = np.sqrt(a.var(ddof=1) / a.size + b.var(ddof=1) / b.size)
    z = 0.0 if se == 0 else (a.mean() - b.mean()) / se
    return float(2 * stats.norm.sf(abs(z)))


def welch_t_pvalue(a, b):
    """Two-sided Welch t-test. Use when n is small (n=30 per policy)."""
    return float(stats.ttest_ind(a, b, equal_var=False).pvalue)


def mean_ci(arr, *, confidence=0.95):
    """Mean and half-width of the Student-t CI. -> (mean, half_width)

    t rather than the 1.96 z-multiplier: at n=30 the normal approximation
    understates the interval by about 4%.
    """
    arr = np.asarray(arr, float)
    half = stats.t.ppf(0.5 + confidence / 2, arr.size - 1) * arr.std(ddof=1) / np.sqrt(arr.size)
    return arr.mean(), half


def pairwise_pvalues(groups, items, test=welch_z_pvalue):
    """Every distinct pair within every group. -> {(group, item_a, item_b): p}

    Only the distinct pairs are returned: the diagonal is trivially p=1 and
    (b, a) merely flips the sign of the statistic, so including them would
    inflate the family size and over-correct.
    """
    return {(g, a, b): test(groups[g][a], groups[g][b])
            for g in groups for a, b in combinations(items, 2)}


def holm(p_raw):
    """Holm-Bonferroni over the whole family. -> {key: corrected p}"""
    m = len(p_raw)
    corrected, running = {}, 0.0
    for rank, key in enumerate(sorted(p_raw, key=p_raw.get)):
        # max() enforces monotonicity along the ranking
        running = max(running, min(1.0, p_raw[key] * (m - rank)))
        corrected[key] = running
    return corrected


def _maximal_cliques(nodes, adjacent):
    """Bron-Kerbosch without pivoting; n is tiny. -> list[frozenset]

    Candidates are iterated in sorted order so the letters are reproducible
    run to run.
    """
    out = []

    def bk(r, p, x):
        if not p and not x:
            out.append(frozenset(r))
            return
        for v in sorted(p):
            bk(r | {v},
               {u for u in p if adjacent(v, u)},
               {u for u in x if adjacent(v, u)})
            p = p - {v}
            x = x | {v}

    bk(set(), set(nodes), set())
    return out


def compact_letters(items, means, significant):
    """Compact Letter Display for one group. -> {item: letters}

    Letters are the maximal cliques of the "not significantly different"
    graph, so two items share a letter exactly when every pairwise test
    between them is non-significant, i.e. they are statistically tied.
    Letter 'a' goes to the clique holding the highest-mean item.
    """
    cliques = _maximal_cliques(items, lambda a, b: a != b and not significant(a, b))
    cliques.sort(key=lambda c: max(means[s] for s in c), reverse=True)

    letters = {s: [] for s in items}
    for idx, clique in enumerate(cliques):
        for s in clique:
            letters[s].append(chr(ord("a") + idx))
    return {s: "".join(sorted(letters[s])) for s in items}


def letters_by_group(groups, items, p_corrected, *, alpha=0.05):
    """Compact Letter Display for every group. -> {group: {item: letters}}"""
    out = {}
    for g in groups:
        means = {s: np.mean(groups[g][s]) for s in items}

        def significant(a, b, _g=g):
            p = p_corrected.get((_g, a, b), p_corrected.get((_g, b, a)))
            return p < alpha

        out[g] = compact_letters(items, means, significant)
    return out


def report(p_raw, p_corrected, *, alpha=0.05):
    """Print family size and which comparisons survive correction."""
    m = len(p_raw)
    print(f"family size m = {m}")
    print(f"raw  p<{alpha}: {sum(p < alpha for p in p_raw.values())} / {m}")
    print(f"Holm p<{alpha}: {sum(p < alpha for p in p_corrected.values())} / {m}")
    survivors = [k for k in p_corrected if p_corrected[k] >= alpha]
    if survivors:
        print(f"\nnot significant after Holm:")
        for k in sorted(survivors, key=lambda k: -p_corrected[k]):
            g, a, b = k
            print(f"  {g:>9}  {a:>15} vs {b:<15} "
                  f"raw={p_raw[k]:.4f}  holm={p_corrected[k]:.4f}")
