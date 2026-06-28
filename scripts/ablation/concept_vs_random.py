import json, numpy as np
from scipy import stats
from statsmodels.stats.multitest import multipletests

d = json.load(open("/data/houwanlong/finllm-mi/outputs/sae/financial_impact.json"))
concepts = list(d["concept_aggregate"].keys())

print("=== CONCEPT vs RANDOM (Paired t-test) ===")
print("{:<20} {:>10} {:>10} {:>10} {:>10} {:>10}".format(
    "Concept", "Concept", "Random", "Delta", "p(paired)", "p(Bonf)"))
print("-" * 72)

results = []
for concept in sorted(concepts):
    c_drops = []; r_drops = []
    for s in d["per_stock"]:
        ci = s.get("concept_impacts", {}).get(concept)
        if ci:
            c_drops.append(ci["acc_drop"])
            r_drops.append(ci["base_acc"] - ci["rand_acc_mean"])
    if len(c_drops) < 3: continue
    c_drops = np.array(c_drops); r_drops = np.array(r_drops)
    delta = np.mean(c_drops - r_drops)
    t, p = stats.ttest_rel(c_drops, r_drops)
    results.append((concept, np.mean(c_drops), np.mean(r_drops), delta, p))
    # Bootstrap CI on delta
    deltas = c_drops - r_drops
    boots = [np.mean(np.random.choice(deltas, len(deltas), replace=True)) for _ in range(10000)]
    ci_lo, ci_hi = np.percentile(boots, 2.5), np.percentile(boots, 97.5)
    sig = ci_lo > 0
    print("{:<20} {:>10.4f} {:>10.4f} {:>+10.4f} {:>10.4f} {:>10}".format(
        concept, np.mean(c_drops), np.mean(r_drops), delta, p, ""))
    print("  CI: [{:+.4f}, {:+.4f}] {}".format(ci_lo, ci_hi, "SIG" if sig else "ns"))

# Bonferroni
pvals = [r[4] for r in results]
rej, p_corr, _, _ = multipletests(pvals, method="bonferroni")
print("\n=== Bonferroni Correction ===")
for i, (concept, cd, rd, delta, p) in enumerate(results):
    print("{}: bonf_p={:.4f} {}".format(concept, p_corr[i], "SIG" if rej[i] else "ns"))
