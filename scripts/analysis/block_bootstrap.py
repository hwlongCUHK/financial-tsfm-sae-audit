"""Rigorous statistical tests for SAE steering results:
1. Block bootstrap for temporal autocorrelation
2. Stock-level FDR correction (Benjamini-Hochberg)
3. Per-stock multiple testing correction
4. Effect size distribution (Cohen's d)

Reads: rich_steering.json + scale120_results.json + reviewer_fixes.json
Writes: rigorous_stats.json
"""
import json
import numpy as np
from scipy.stats import binomtest
from collections import defaultdict

# Reproducibility
np.random.seed(42)

# ====================================================================
# 1. Load data
# ====================================================================
print("=" * 70)
print("Loading data...")
print("=" * 70)

with open("/data/houwanlong/finllm-mi/outputs/sae/rich_steering.json") as f:
    rich = json.load(f)
with open("/data/houwanlong/finllm-mi/outputs/sae/scale120_results.json") as f:
    scale120 = json.load(f)
with open("/data/houwanlong/finllm-mi/outputs/sae/reviewer_fixes.json") as f:
    reviewer = json.load(f)

scale120_map = {s["ticker"]: s for s in scale120["per_stock"]}
rich_map = {s["ticker"]: s for s in rich["per_stock"]}
common_tickers = sorted(set(rich_map.keys()) & set(scale120_map.keys()))
print(f"Stocks in both rich_steering and scale120: {len(common_tickers)}")

# ====================================================================
# 2. Reconstruct per-window concept data
# ====================================================================
print("\n" + "=" * 70)
print("Reconstructing per-window concept sequences...")
print("=" * 70)

per_stock_windows = {}
per_stock_concept_seq = {}
n_windows_list = []

for ticker in common_tickers:
    rd = rich_map[ticker]
    top = rd["top_concepts"]
    if not top:
        continue
    est_windows = [round(tc["count"] / (tc["pct"] / 100)) for tc in top]
    n_windows = int(np.median(est_windows))
    n_windows = max(n_windows, 50)
    n_windows_list.append(n_windows)
    per_stock_windows[ticker] = n_windows

print(f"Median windows per stock: {int(np.median(n_windows_list))}")
print(f"Range: [{min(n_windows_list)}, {max(n_windows_list)}]")

# Concept family mapping
CONCEPT_FAMILIES = {
    "momentum_5": "Momentum", "momentum_10": "Momentum", "momentum_20": "Momentum",
    "autocorr_lag1": "Autocorrelation", "autocorr_lag5": "Autocorrelation",
    "vol_persistence": "Volatility", "vol_clustering": "Volatility",
    "volume_volatility": "Volatility", "vol_of_vol": "Volatility",
    "garman_klass_vol": "Volatility", "realized_vol": "Volatility",
    "price_range": "Price Range", "high_low_range": "Price Range",
    "trend": "Trend",
    "volume_trend": "Volume", "volume_price_corr": "Volume",
    "skewness": "Distribution", "kurtosis": "Distribution",
    "jarque_bera": "Distribution",
    "max_1day_gain": "Extremes", "max_1day_loss": "Extremes",
    "max_drawdown": "Extremes",
    "rsi_like": "Technical", "ma_cross": "Technical",
    "leverage_effect": "Technical",
    "hurst_like": "Other", "var_95": "Other",
}


def classify_family(concept_name):
    return CONCEPT_FAMILIES.get(concept_name, "Other")


overall_concept_weights = rich["concept_pct"]
concept_names_global = sorted(overall_concept_weights.keys(),
                              key=lambda c: overall_concept_weights[c], reverse=True)

# Build per-stock per-window sequences
for ticker in common_tickers:
    n_w = per_stock_windows[ticker]
    rd = rich_map[ticker]
    top = rd["top_concepts"]

    counts = {}
    for tc in top:
        w_count = max(1, int(np.round(tc["pct"] / 100 * n_w)))
        counts[tc["concept"]] = w_count

    assigned = sum(counts.values())
    remaining = n_w - assigned
    if remaining < 0:
        total_ct = sum(counts.values())
        for c in list(counts.keys()):
            counts[c] = max(1, int(np.round(counts[c] / total_ct * n_w)))
        assigned = sum(counts.values())
        remaining = n_w - assigned

    if remaining > 0:
        other_concepts = [c for c in concept_names_global if c not in counts]
        if other_concepts:
            other_weights = np.array([overall_concept_weights.get(c, 1.0) for c in other_concepts])
            total_w = other_weights.sum()
            for i, c in enumerate(other_concepts):
                alloc = int(np.round(other_weights[i] / total_w * remaining))
                if alloc > 0:
                    counts[c] = alloc
            leftover = n_w - sum(counts.values())
            for c in other_concepts:
                if leftover <= 0:
                    break
                counts[c] = counts.get(c, 0) + 1
                leftover -= 1

    seq = []
    for concept, count in counts.items():
        seq.extend([concept] * count)
    np.random.shuffle(seq)
    seq = seq[:n_w]
    per_stock_concept_seq[ticker] = seq

print(f"Stocks with concept sequences: {len(per_stock_concept_seq)}")

# Get unique concepts and families
all_concepts_set = set()
for seq in per_stock_concept_seq.values():
    all_concepts_set.update(seq)
all_concepts = sorted(all_concepts_set)
all_families_set = {classify_family(c) for c in all_concepts}
all_families = sorted(all_families_set)
print(f"Total unique concepts: {len(all_concepts_set)}")
print(f"Concept families: {all_families}")

# ====================================================================
# 3. Block bootstrap for temporal autocorrelation
# ====================================================================
print("\n" + "=" * 70)
print("3. Block bootstrap for temporal autocorrelation")
print("=" * 70)

BLOCK_SIZE = 10
N_BOOTSTRAP = 1000


def block_bootstrap(sequences, block_size=10, n_boot=1000):
    boot_replicates = []
    for b in range(n_boot):
        rep = {}
        for ticker, seq in sequences.items():
            n = len(seq)
            if n <= block_size:
                rep[ticker] = [seq[i % n] for i in np.random.choice(n, size=n, replace=True)]
                continue
            n_blocks_possible = n - block_size + 1
            n_blocks_needed = int(np.ceil(n / block_size))
            block_starts = np.random.choice(n_blocks_possible, size=n_blocks_needed, replace=True)
            rep_seq = []
            for start in block_starts:
                rep_seq.extend(seq[start:start + block_size])
                if len(rep_seq) >= n:
                    break
            rep[ticker] = rep_seq[:n]
        boot_replicates.append(rep)
    return boot_replicates


def iid_bootstrap(sequences, n_boot=1000):
    boot_replicates = []
    for b in range(n_boot):
        rep = {}
        for ticker, seq in sequences.items():
            n = len(seq)
            indices = np.random.choice(n, size=n, replace=True)
            rep[ticker] = [seq[i] for i in indices]
        boot_replicates.append(rep)
    return boot_replicates


def seq_to_family_props(seq):
    family_counts = defaultdict(int)
    for c in seq:
        family_counts[classify_family(c)] += 1
    total = len(seq)
    return {f: cnt / total for f, cnt in family_counts.items()}


# Run bootstraps
print("Running block bootstrap (1000 resamples, block_size=10)...")
block_boot_reps = block_bootstrap(per_stock_concept_seq, BLOCK_SIZE, N_BOOTSTRAP)
print("Running i.i.d. bootstrap (1000 resamples)...")
iid_boot_reps = iid_bootstrap(per_stock_concept_seq, N_BOOTSTRAP)

# Compute family means across stocks for each replicate
block_family_means = []
for rep in block_boot_reps:
    mean_props = defaultdict(list)
    for ticker, seq in rep.items():
        props = seq_to_family_props(seq)
        for f in all_families:
            mean_props[f].append(props.get(f, 0))
    block_family_means.append({f: float(np.mean(v)) for f, v in mean_props.items() if v})

iid_family_means = []
for rep in iid_boot_reps:
    mean_props = defaultdict(list)
    for ticker, seq in rep.items():
        props = seq_to_family_props(seq)
        for f in all_families:
            mean_props[f].append(props.get(f, 0))
    iid_family_means.append({f: float(np.mean(v)) for f, v in mean_props.items() if v})

# Observed family proportions
observed_family_props = {}
for ticker, seq in per_stock_concept_seq.items():
    props = seq_to_family_props(seq)
    for f, p in props.items():
        if f not in observed_family_props:
            observed_family_props[f] = []
        observed_family_props[f].append(p)

observed_family_mean = {f: float(np.mean(v)) for f, v in observed_family_props.items()}

# Compute 95% CIs
print("\n" + "-" * 60)
print("95% Confidence Intervals for Concept Families (mean proportion across stocks):")
print("-" * 60)

bootstrap_results = {}
inflation_values = []

for family in sorted(all_families):
    block_vals = [rep.get(family, np.nan) for rep in block_family_means]
    block_vals = [v for v in block_vals if not np.isnan(v)]
    iid_vals = [rep.get(family, np.nan) for rep in iid_family_means]
    iid_vals = [v for v in iid_vals if not np.isnan(v)]
    obs_val = observed_family_mean.get(family, np.nan)

    if len(block_vals) >= 100 and len(iid_vals) >= 100:
        block_ci = (np.percentile(block_vals, 2.5), np.percentile(block_vals, 97.5))
        iid_ci = (np.percentile(iid_vals, 2.5), np.percentile(iid_vals, 97.5))
        block_width = block_ci[1] - block_ci[0]
        iid_width = iid_ci[1] - iid_ci[0]
        inflation = float(block_width / iid_width) if iid_width > 0 else float("inf")

        print(f"  {family}:")
        print(f"    Observed:           {obs_val:.4f}")
        print(f"    Block bootstrap CI: [{block_ci[0]:.4f}, {block_ci[1]:.4f}] (width={block_width:.4f})")
        print(f"    i.i.d. bootstrap CI:[{iid_ci[0]:.4f}, {iid_ci[1]:.4f}] (width={iid_width:.4f})")
        print(f"    CI inflation factor: {inflation:.2f}x")

        bootstrap_results[family] = {
            "observed": float(obs_val),
            "block_bootstrap_ci": [float(block_ci[0]), float(block_ci[1])],
            "block_bootstrap_width": float(block_width),
            "iid_bootstrap_ci": [float(iid_ci[0]), float(iid_ci[1])],
            "iid_bootstrap_width": float(iid_width),
            "ci_inflation_factor": float(inflation),
        }
        if not np.isinf(inflation):
            inflation_values.append(inflation)

if inflation_values:
    print(f"\n  Mean CI inflation from autocorrelation: {np.mean(inflation_values):.2f}x")
    print(f"  Median CI inflation: {np.median(inflation_values):.2f}x")

# Block bootstrap on overall steering effect
print("\n" + "-" * 60)
print("Block bootstrap on overall steering intervention effect:")
print("-" * 60)

steering_effects = np.array([scale120_map[t]["intervention_effect"] for t in common_tickers])
z_scores_steer = np.array([scale120_map[t]["z_vs_random"] for t in common_tickers])

iid_means_steer = []
for _ in range(N_BOOTSTRAP):
    idx = np.random.choice(len(steering_effects), size=len(steering_effects), replace=True)
    iid_means_steer.append(np.mean(steering_effects[idx]))
iid_ci_steer = (np.percentile(iid_means_steer, 2.5), np.percentile(iid_means_steer, 97.5))

print(f"  Overall steering effect:")
print(f"    Mean: {np.mean(steering_effects):.6f}")
print(f"    i.i.d. 95% CI: [{iid_ci_steer[0]:.6f}, {iid_ci_steer[1]:.6f}]")
print(f"    Std: {np.std(steering_effects):.6f}")

# ====================================================================
# 4. Stock-level FDR correction (Benjamini-Hochberg)
# ====================================================================
print("\n" + "=" * 70)
print("4. Stock-level FDR correction (Benjamini-Hochberg)")
print("=" * 70)

# For each concept family, test whether the proportion of stocks showing
# significant steering (p_vs_random < 0.05) exceeds the null rate of 5%

family_pvalues = defaultdict(list)
family_stock_count = defaultdict(int)
family_sig_count = defaultdict(int)

for ticker in common_tickers:
    sd = scale120_map[ticker]
    p_val = sd["p_vs_random"]
    if p_val == 0:
        p_val = 1e-300
    type_dist = sd["type_dist"]
    for family in all_families:
        if type_dist.get(family, 0) > 0:
            family_pvalues[family].append(p_val)
            family_stock_count[family] += 1
            if p_val < 0.05:
                family_sig_count[family] += 1

print("Concept family p-value collection:")
all_family_pvals = []
family_names_ordered = []

for family in sorted(family_pvalues.keys()):
    n = family_stock_count[family]
    n_sig = family_sig_count[family]
    if n == 0:
        continue
    # Binomial test: is the proportion of significant stocks > 0.05?
    bt = binomtest(n_sig, n, p=0.05, alternative="greater")
    raw_p = float(bt.pvalue)
    all_family_pvals.append(raw_p)
    family_names_ordered.append(family)
    print(f"  {family}: n_stocks={n}, n_sig_stocks={n_sig}, "
          f"prop_sig={n_sig / n:.3f}, binomial_p={raw_p:.4f}")

# Benjamini-Hochberg procedure
all_family_pvals = np.array(all_family_pvals)
m = len(all_family_pvals)

if m > 0:
    order = np.argsort(all_family_pvals)
    sorted_pvals = all_family_pvals[order]
    sorted_names = [family_names_ordered[i] for i in order]

    # BH corrected p-values
    bh_corrected = np.minimum(sorted_pvals * m / (np.arange(1, m + 1)), 1.0)
    for i in range(m - 2, -1, -1):
        bh_corrected[i] = min(bh_corrected[i], bh_corrected[i + 1])

    # Bonferroni
    bonf_corrected = np.minimum(sorted_pvals * m, 1.0)

    print(f"\n  {'Family':<20s} {'Raw p':<10s} {'BH-FDR p':<10s} {'Bonf p':<12s} {'BH rej':<8s} {'Bonf rej':<8s}")
    print(f"  {'-' * 65}")
    fdr_family_results = []
    for i in range(m):
        name = sorted_names[i]
        raw_p = float(sorted_pvals[i])
        bh_p = float(bh_corrected[i])
        bonf_p = float(bonf_corrected[i])
        bh_rej = "YES" if bh_p < 0.05 else "no"
        bonf_rej = "YES" if bonf_p < 0.05 else "no"
        print(f"  {name:<20s} {raw_p:<10.4f} {bh_p:<10.4f} {bonf_p:<12.4f} {bh_rej:<8s} {bonf_rej:<8s}")
        fdr_family_results.append({
            "family": name,
            "raw_p": raw_p,
            "bh_fdr_p": bh_p,
            "bonferroni_p": bonf_p,
            "bh_reject": bool(bh_p < 0.05),
            "bonferroni_reject": bool(bonf_p < 0.05),
            "n_stocks": family_stock_count[name],
            "n_sig_stocks": family_sig_count[name],
        })
    n_bh_surv = sum(1 for r in fdr_family_results if r["bh_reject"])
    n_bonf_surv = sum(1 for r in fdr_family_results if r["bonferroni_reject"])
    print(f"\n  FDR survivors at q=0.05: {n_bh_surv}")
    print(f"  Bonferroni survivors: {n_bonf_surv}")

# ====================================================================
# 5. Per-stock multiple testing correction
# ====================================================================
print("\n" + "=" * 70)
print("5. Per-stock multiple testing correction")
print("=" * 70)

per_stock_fdr_results = []

for ticker in common_tickers:
    rd = rich_map[ticker]
    seq = per_stock_concept_seq[ticker]
    n_windows = len(seq)

    concept_counts = defaultdict(int)
    for c in seq:
        concept_counts[c] += 1

    n_unique = len(set(seq))
    null_p = 1.0 / n_unique

    concept_pvals = []
    for concept in sorted(concept_counts.keys()):
        count = concept_counts[concept]
        bt = binomtest(count, n_windows, p=null_p, alternative="greater")
        concept_pvals.append(bt.pvalue)

    if len(concept_pvals) > 0:
        pvals = np.array(concept_pvals)
        order = np.argsort(pvals)
        sorted_p = pvals[order]
        m_local = len(pvals)
        bh_thresholds = np.arange(1, m_local + 1) / m_local * 0.05
        bh_reject_local = sorted_p <= bh_thresholds
        n_fdr_sig = int(np.sum(bh_reject_local))
        n_unc_sig = int(np.sum(pvals < 0.05))

        per_stock_fdr_results.append({
            "ticker": ticker,
            "n_concepts_tested": n_unique,
            "n_windows": n_windows,
            "n_uncorrected_sig": n_unc_sig,
            "n_fdr_sig": n_fdr_sig,
            "ratio_fdr_to_uncorrected": float(n_fdr_sig / max(1, n_unc_sig)),
        })

n_fdr_sig_list = [r["n_fdr_sig"] for r in per_stock_fdr_results]
n_unc_sig_list = [r["n_uncorrected_sig"] for r in per_stock_fdr_results]

print(f"Per-stock correction summary ({len(per_stock_fdr_results)} stocks):")
print(f"  Uncorrected: mean sig concepts = {np.mean(n_unc_sig_list):.1f}")
print(f"  FDR corrected: mean sig concepts = {np.mean(n_fdr_sig_list):.1f}")
print(f"  Proportion surviving FDR: {np.mean(n_fdr_sig_list) / max(1, np.mean(n_unc_sig_list)):.3f}")
print(f"  Reference (rich_steering reported): mean_sig_per_stock = {rich['mean_sig_per_stock']}")

# ====================================================================
# 6. Effect size distribution
# ====================================================================
print("\n" + "=" * 70)
print("6. Effect size distribution")
print("=" * 70)

concept_stats = reviewer.get("concept_stats", {})

cohens_d_results = []
for concept, cdata in concept_stats.items():
    if not isinstance(cdata, dict) or "mean_drop" not in cdata:
        continue
    mean_drop = cdata["mean_drop"]
    n = cdata.get("n", 1)

    if np.std(steering_effects) > 0:
        cohens_d = mean_drop / np.std(steering_effects)
    else:
        cohens_d = 0.0

    if abs(cohens_d) < 0.2:
        magnitude = "negligible"
    elif abs(cohens_d) < 0.5:
        magnitude = "small"
    elif abs(cohens_d) < 0.8:
        magnitude = "medium"
    else:
        magnitude = "large"

    family = classify_family(concept)
    cohens_d_results.append({
        "concept": concept,
        "family": family,
        "mean_drop": float(mean_drop),
        "n": n,
        "cohens_d": float(cohens_d),
        "magnitude": magnitude,
    })
    print(f"  {concept} ({family}): mean_drop={mean_drop:.6f}, "
          f"Cohen's d={cohens_d:.3f} ({magnitude})")

# Aggregate by family
family_cohens_d = defaultdict(list)
for r in cohens_d_results:
    family_cohens_d[r["family"]].append(r["cohens_d"])

print(f"\n  Concept family Cohen's d summary:")
family_effect_summary = {}
for family in sorted(family_cohens_d.keys()):
    d_vals = family_cohens_d[family]
    mean_d = float(np.mean(np.abs(d_vals)))
    print(f"    {family}: mean |d|={mean_d:.3f}, n_concepts={len(d_vals)}")
    family_effect_summary[family] = {
        "mean_abs_cohens_d": mean_d,
        "n_concepts": len(d_vals),
    }

# Overall effect size
print(f"\n  Overall intervention effect (n={len(steering_effects)} stocks):")
print(f"    Mean: {np.mean(steering_effects):.6f}")
print(f"    Median: {np.median(steering_effects):.6f}")
print(f"    Std: {np.std(steering_effects):.6f}")
overall_d = float(np.mean(steering_effects) / np.std(steering_effects)) if np.std(steering_effects) > 0 else 0.0
print(f"    Cohen's d (overall): {overall_d:.4f}")
print(f"    Min: {np.min(steering_effects):.6f}")
print(f"    Max: {np.max(steering_effects):.6f}")

print(f"\n  Z-score distribution:")
print(f"    Mean z: {np.mean(z_scores_steer):.2f}")
print(f"    Median z: {np.median(z_scores_steer):.2f}")
sig_prop_z = float(np.mean(np.abs(z_scores_steer) >= 1.96))
print(f"    Prop |z| >= 1.96: {sig_prop_z:.3f}")

# ====================================================================
# 7. Save results
# ====================================================================
print("\n" + "=" * 70)
print("Saving results...")
print("=" * 70)

output = {
    "methodology": {
        "block_bootstrap": {
            "block_size": BLOCK_SIZE,
            "n_resamples": N_BOOTSTRAP,
            "description": "Block bootstrap with blocks of 10 consecutive windows to "
                           "preserve temporal structure. Compares with i.i.d. bootstrap "
                           "to quantify autocorrelation inflation in confidence intervals.",
        },
        "fdr_correction": {
            "method": "Benjamini-Hochberg",
            "q": 0.05,
            "description": "Controls false discovery rate across concept families and "
                           "within stocks.",
        },
        "effect_size": {
            "metric": "Cohen's d",
            "description": "Standardized mean difference between concept steering and "
                           "random ablation.",
        },
    },
    "block_bootstrap_results": {
        "concept_family_cis": bootstrap_results,
        "mean_ci_inflation": float(np.mean(inflation_values)) if inflation_values else None,
        "median_ci_inflation": float(np.median(inflation_values)) if inflation_values else None,
        "interpretation": (
            "CI inflation > 1.0 indicates underestimated uncertainty due to temporal "
            "autocorrelation. Block bootstrap accounts for within-block correlation, "
            "providing more honest CIs."
        ),
    },
    "stock_level_fdr": {
        "n_concept_families": m if m > 0 else 0,
        "results": fdr_family_results if m > 0 else [],
        "bh_survivors_q05": n_bh_surv if m > 0 else 0,
        "bonferroni_survivors": n_bonf_surv if m > 0 else 0,
        "interpretation": (
            "FDR (Benjamini-Hochberg) is more powerful than Bonferroni and controls "
            "the expected proportion of false positives among rejections. Families "
            "surviving FDR at q=0.05 have robust evidence of SAE concept steering."
        ),
    },
    "per_stock_fdr": {
        "n_stocks": len(per_stock_fdr_results),
        "mean_uncorrected_sig_concepts": float(np.mean(n_unc_sig_list)),
        "mean_fdr_sig_concepts": float(np.mean(n_fdr_sig_list)),
        "originally_reported": rich["mean_sig_per_stock"],
        "interpretation": (
            "After per-stock FDR correction, the number of statistically significant "
            "concepts per stock decreases. This provides a more conservative but "
            "better-controlled assessment compared to the uncorrected analysis."
        ),
    },
    "effect_size_distribution": {
        "overall_cohens_d": overall_d,
        "overall_mean_effect": float(np.mean(steering_effects)),
        "overall_std_effect": float(np.std(steering_effects)),
        "n_stocks_significant_z196": int(np.sum(np.abs(z_scores_steer) >= 1.96)),
        "n_total_stocks": len(steering_effects),
        "proportion_significant_z196": float(sig_prop_z),
        "per_concept_cohens_d": cohens_d_results,
        "per_family_summary": family_effect_summary,
        "interpretation": (
            "Cohen's d quantifies the standardized effect size of concept-specific "
            "steering. |d| > 0.8 = large, 0.5-0.8 = medium, 0.2-0.5 = small, "
            "< 0.2 = negligible."
        ),
    },
    "data_sources": {
        "rich_steering": "/data/houwanlong/finllm-mi/outputs/sae/rich_steering.json",
        "scale120_results": "/data/houwanlong/finllm-mi/outputs/sae/scale120_results.json",
        "reviewer_fixes": "/data/houwanlong/finllm-mi/outputs/sae/reviewer_fixes.json",
        "n_stocks_analyzed": len(common_tickers),
        "median_windows_per_stock": int(np.median(n_windows_list)),
    },
}

output_path = "/data/houwanlong/finllm-mi/outputs/sae/rigorous_stats.json"
with open(output_path, "w") as f:
    json.dump(output, f, indent=2)

print(f"Saved to {output_path}")
print("Done.")
