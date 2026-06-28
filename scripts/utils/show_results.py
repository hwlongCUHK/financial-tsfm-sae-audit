import json

with open("/data/houwanlong/finllm-mi/outputs/sae/baseline_comparison.json") as f:
    data = json.load(f)

print("=" * 70)
print("SAE vs PCA vs Random - BASELINE COMPARISON")
print("=" * 70)
print("\n20 stocks (5/sector), Layer 6, k=64, 3000 steps\n")

for exp_key in ["exp2x", "exp4x", "exp8x"]:
    exp = exp_key.replace("exp", "").replace("x", "x")
    a = data["aggregate"][exp_key]
    s, p, r = a["sae"], a["pca"], a["random"]
    
    print(f"--- SAE {exp} ---")
    print(f"{'Metric':<25} {'SAE':<20} {'PCA':<20} {'Random':<20}")
    print("-" * 85)
    
    print(f"{'Max |r|':<25} {s['max_corr']['mean']:.4f} +- {s['max_corr']['std']:.3f}        {p['max_corr']['mean']:.4f} +- {p['max_corr']['std']:.3f}        {r['max_corr']['mean']:.4f} +- {r['max_corr']['std']:.3f}")
    print(f"{'Mean |r|':<25} {s['mean_corr']['mean']:.4f} +- {s['mean_corr']['std']:.3f}        {p['mean_corr']['mean']:.4f} +- {p['mean_corr']['std']:.3f}        {r['mean_corr']['mean']:.4f} +- {r['mean_corr']['std']:.3f}")
    print(f"{'% > 0.3':<25} {s['pct_above_threshold']['mean']:.1f}%                      {p['pct_above_threshold']['mean']:.1f}%                  {r['pct_above_threshold']['mean']:.1f}%")
    print(f"{'Var Explained':<25} {s['var_explained']['mean']:.4f}              {p['cum_var']['mean']:.4f}              N/A")
    print(f"{'Ablation Delta':<25} {s['delta_corr']['mean']:.4f} +- {s['delta_corr']['std']:.4f}     {p['delta']['mean']:.4f} +- {p['delta']['std']:.4f}        {r['delta']['mean']:.4f} +- {r['delta']['std']:.4f}")
    print()

print("=" * 70)
print("STATISTICAL TESTS (paired t-test, SAE 4x vs baselines)")
print("=" * 70)
for name, t in data["stats_tests"].items():
    sig = "***" if t["p"] < 0.001 else ("**" if t["p"] < 0.01 else ("*" if t["p"] < 0.05 else "ns"))
    print(f"  {name}: t={t['t']:.3f}, p={t['p']:.6f} {sig}")
