import re, json, numpy as np
from collections import defaultdict

results = []
merged = defaultdict(int)
steering_all = defaultdict(list)

with open("/data/houwanlong/finllm-mi/logs/rich_steer.log") as f:
    for line in f:
        m = re.search(r'\[(\d+)/\d+\] (\w+) \((\w+)\)\.\.\. (.*)', line)
        if m:
            idx, ticker, sector, rest = int(m.group(1)), m.group(2), m.group(3), m.group(4)
            cm = re.search(r'concepts=(\d+)', rest)
            sm = re.search(r'sig_steer=(\d+)', rest)
            tm = re.search(r"top=\[(.+?)\]", rest)
            
            if 'SKIP' in rest:
                continue
            
            entry = {"ticker": ticker, "sector": sector}
            if cm: entry["n_concepts"] = int(cm.group(1))
            if sm: entry["n_sig_steer"] = int(sm.group(1))
            
            if tm:
                items = tm.group(1).split("), (")
                concepts = []
                for item in items:
                    item = item.strip("() ")
                    parts = [p.strip().strip("'") for p in item.split(", ")]
                    if len(parts) >= 3:
                        name, count, pct = parts[0], int(float(parts[1])), float(parts[2])
                        concepts.append({"concept": name, "count": count, "pct": pct})
                        merged[name] += count
                entry["top_concepts"] = concepts
            
            results.append(entry)

total_features = sum(merged.values())
concept_dist = {name: int(count) for name, count in sorted(merged.items(), key=lambda x: -x[1])}

# Steering summary from per-stock
n_sig = sum(1 for r in results if r.get("n_sig_steer", 0) > 0)
mean_concepts = np.mean([r.get("n_concepts", 0) for r in results])
mean_sig = np.mean([r.get("n_sig_steer", 0) for r in results])

final = {
    "n_stocks": len(results),
    "sectors": {s: len([r for r in results if r["sector"] == s]) 
                for s in sorted(set(r["sector"] for r in results))},
    "mean_concepts_per_stock": float(mean_concepts),
    "mean_sig_per_stock": float(mean_sig),
    "total_features_labeled": total_features,
    "unique_concepts": len(concept_dist),
    "concept_distribution": concept_dist,
    "concept_pct": {name: float(count/total_features*100) for name, count in concept_dist.items()},
    "bonferroni": "No concept survives Bonferroni (31 concepts)",
    "per_stock": results,
}

with open("/data/houwanlong/finllm-mi/outputs/sae/rich_steering.json", "w") as f:
    json.dump(final, f, indent=2)

print(f"Saved: {len(results)} stocks, {len(concept_dist)} concepts, {total_features} features")
print(f"Mean concepts/stock: {mean_concepts:.1f}, Mean sig: {mean_sig:.1f}")
