"""SAE feature interpretability: label features with financial concepts."""
import torch
import numpy as np
import json, sys, time, argparse
from pathlib import Path
import pandas as pd
from collections import defaultdict
from sklearn.cluster import KMeans
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import f1_score

sys.path.insert(0, "/data/houwanlong/finllm-mi/code")
from model.kronos import Kronos, KronosTokenizer
from safetensors.torch import load_file

# Reuse the SAE class from training
class TopKSAE(torch.nn.Module):
    def __init__(self, d_model, d_hidden, k=64):
        super().__init__()
        self.d_model = d_model
        self.d_hidden = d_hidden
        self.k = k
        self.encoder = torch.nn.Linear(d_model, d_hidden, bias=True)
        self.decoder = torch.nn.Linear(d_hidden, d_model, bias=False)
        self.b_pre = torch.nn.Parameter(torch.zeros(d_model))
    def forward(self, x):
        x_centered = x - self.b_pre
        latents = self.encoder(x_centered)
        topk_vals, topk_idx = torch.topk(latents, self.k, dim=-1)
        mask = torch.zeros_like(latents)
        mask.scatter_(-1, topk_idx, 1.0)
        latents = latents * mask
        x_recon = self.decoder(latents) + self.b_pre
        return x_recon, latents


def load_data_and_model(device):
    """Load Kronos + tokenizer and K-line data."""
    tokenizer = KronosTokenizer.from_pretrained("/data/houwanlong/models/Kronos-Tokenizer-base").to(device).eval()
    with open("/data/houwanlong/models/Kronos-base/config.json") as f:
        cfg = json.load(f)
    model = Kronos(
        s1_bits=cfg["s1_bits"], s2_bits=cfg["s2_bits"],
        n_layers=cfg["n_layers"], d_model=cfg["d_model"],
        n_heads=cfg["n_heads"], ff_dim=cfg["ff_dim"],
        ffn_dropout_p=cfg["ffn_dropout_p"], attn_dropout_p=cfg["attn_dropout_p"],
        resid_dropout_p=cfg["resid_dropout_p"], token_dropout_p=cfg["token_dropout_p"],
        learn_te=cfg["learn_te"],
    )
    state_dict = load_file("/data/houwanlong/models/Kronos-base/model.safetensors")
    model.load_state_dict(state_dict, strict=False)
    model = model.to(device).half().eval()

    # Load data
    df = pd.read_csv("/data/houwanlong/finllm-mi/code/finetune_csv/data/HK_ali_09988_kline_5min_all.csv")
    for col in ["open","close","high","low","volume","amount"]:
        if col not in df.columns: df[col] = 0.0
    data = df[["open","close","high","low","volume","amount"]].values.astype(np.float32)
    data = data[~np.isnan(data).any(axis=1)]
    mn, st = data.mean(0), data.std(0)
    data_norm = np.clip((data - mn) / (st + 1e-5), -5, 5)
    return tokenizer, model, data_norm


def compute_financial_labels(data_norm, lookback=64, stride=32):
    """Compute per-window financial property labels."""
    windows = []
    labels = []
    for i in range(0, len(data_norm) - lookback, stride):
        win = data_norm[i:i+lookback]
        windows.append(win)
        close = win[:, 1]
        returns = np.diff(close) / (close[:-1] + 1e-5)

        # Financial properties
        vol = np.std(returns)
        slope = np.polyfit(np.arange(lookback), close, 1)[0]
        max_dd = np.min(close / np.maximum.accumulate(close) - 1)
        price_range = (close.max() - close.min()) / close.mean()
        vol_cluster = np.mean(returns**2) / (np.std(returns)**2 + 1e-5)  # >1 means vol clustering
        skew = np.mean((returns - returns.mean())**3) / (returns.std()**3 + 1e-5)
        kurtosis = np.mean((returns - returns.mean())**4) / (returns.std()**4 + 1e-5)

        labels.append({
            "vol": float(vol),
            "trend": float(slope),
            "max_drawdown": float(max_dd),
            "price_range": float(price_range),
            "vol_clustering": float(vol_cluster),
            "skewness": float(skew),
            "kurtosis": float(kurtosis),
        })
    return windows, labels


def interpret_layer(sae, acts, labels, device):
    """Interpret SAE features for one layer."""
    sae.eval()
    d_model = sae.d_model
    d_hidden = sae.d_hidden

    # Get feature activations for all samples
    acts_tensor = torch.from_numpy(acts).float()
    n_samples = acts_tensor.shape[0]

    # Process in batches
    all_latents = []
    batch_size = 256
    with torch.no_grad():
        for i in range(0, n_samples, batch_size):
            batch = acts_tensor[i:i+batch_size].to(device)
            _, latents = sae(batch)
            all_latents.append(latents.cpu().numpy())

    all_latents = np.concatenate(all_latents, axis=0)  # (n_samples, d_hidden)

    # Only analyze alive features
    feat_usage = (all_latents != 0).sum(axis=0)
    alive_mask = feat_usage > 10  # at least 10 activations
    alive_idx = np.where(alive_mask)[0]
    n_alive = len(alive_idx)
    print(f"  Alive features: {n_alive}/{d_hidden} ({n_alive/d_hidden:.1%})")

    if n_alive == 0:
        return {}

    alive_latents = all_latents[:, alive_idx]  # (n_samples, n_alive)

    # Match sample count with labels
    n = min(n_samples, len(labels))
    alive_latents = alive_latents[:n]
    labels = labels[:n]

    # Compute per-feature financial correlations
    label_keys = ["vol", "trend", "max_drawdown", "price_range", "vol_clustering", "skewness", "kurtosis"]
    label_names = ["Volatility", "Trend", "Max Drawdown", "Price Range", "Vol Clustering", "Skewness", "Kurtosis"]
    label_arr = np.array([[l[k] for k in label_keys] for l in labels])  # (n, 7)

    # Correlation between each feature and each label
    feature_corrs = np.zeros((n_alive, len(label_keys)))
    for j in range(n_alive):
        feat_vals = alive_latents[:, j]
        active = feat_vals != 0
        if active.sum() < 5:
            continue
        for k in range(len(label_keys)):
            corr = np.corrcoef(feat_vals[active], label_arr[active, k])[0, 1]
            feature_corrs[j, k] = corr if not np.isnan(corr) else 0

    # Classify each feature by its highest-correlated label
    feature_types = []
    for j in range(n_alive):
        best_k = np.argmax(np.abs(feature_corrs[j]))
        corr_val = feature_corrs[j, best_k]
        direction = "positive" if feature_corrs[j, best_k] > 0 else "negative"
        feature_types.append({
            "feature_id": int(alive_idx[j]),
            "type": label_keys[best_k],
            "type_name": label_names[best_k],
            "correlation": float(corr_val),
            "direction": direction,
            "activation_count": int(feat_usage[alive_idx[j]]),
        })

    # Cluster features by their activation patterns
    n_clusters = min(8, n_alive // 10)
    if n_clusters >= 2:
        # Use feature correlation profiles as clustering input
        scaler = StandardScaler()
        corr_scaled = scaler.fit_transform(feature_corrs)
        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        clusters = kmeans.fit_predict(corr_scaled)

        # Label each cluster by its dominant label
        cluster_labels = {}
        for c in range(n_clusters):
            mask = clusters == c
            if mask.sum() == 0:
                continue
            cluster_corrs = feature_corrs[mask].mean(axis=0)
            dominant = np.argmax(np.abs(cluster_corrs))
            cluster_labels[c] = {
                "name": label_names[dominant],
                "key": label_keys[dominant],
                "size": int(mask.sum()),
                "mean_correlation": float(cluster_corrs[dominant]),
            }
    else:
        clusters = np.zeros(n_alive, dtype=int)
        cluster_labels = {0: {"name": "Mixed", "key": "mixed", "size": n_alive, "mean_correlation": 0}}

    # Count features per type
    type_counts = defaultdict(int)
    for ft in feature_types:
        type_counts[ft["type_name"]] += 1

    # Find top features per type
    top_features = {}
    for type_name in label_names:
        typed = [ft for ft in feature_types if ft["type_name"] == type_name]
        typed.sort(key=lambda x: abs(x["correlation"]), reverse=True)
        top_features[type_name] = typed[:5]

    return {
        "n_alive": n_alive,
        "n_total": d_hidden,
        "type_distribution": dict(type_counts),
        "clusters": {str(k): v for k, v in cluster_labels.items()},
        "top_features_by_type": {k: v for k, v in top_features.items() if v},
        "feature_correlation_range": {
            "max_abs_corr": float(np.abs(feature_corrs).max()),
            "mean_abs_corr": float(np.abs(feature_corrs).mean()),
            "n_high_corr": int((np.abs(feature_corrs).max(axis=1) > 0.3).sum()),
        },
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sae-dir", default="/data/houwanlong/finllm-mi/outputs/sae")
    parser.add_argument("--output", default="/data/houwanlong/finllm-mi/outputs/sae/interpretability.json")
    parser.add_argument("--layers", type=str, default="0,3,6,9,11")
    parser.add_argument("--n-samples", type=int, default=5000)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()

    device = args.device
    t0 = time.time()

    # Load data and model
    print("Loading models and data...")
    tokenizer, model, data_norm = load_data_and_model(device)

    # Compute financial labels
    print("Computing financial labels...")
    windows, labels = compute_financial_labels(data_norm, lookback=64, stride=32)
    print(f"  {len(windows)} windows, {len(labels)} labels")

    # Extract activations for selected layers
    layers = [int(x) for x in args.layers.split(",")]
    d_model = 832
    d_hidden = d_model * 4  # expansion=4 from training

    print(f"\nExtracting activations for layers {layers}...")
    acts = {}
    n_layers = len(model.transformer)
    all_acts_raw = {l: [] for l in layers}

    def make_hook(layer_idx):
        def h(module, input, output):
            act = output[0] if isinstance(output, tuple) else output
            all_acts_raw[layer_idx].append(act[:, -1, :].detach().cpu().float().numpy())
        return h

    hooks = []
    for l in layers:
        if l < n_layers:
            hooks.append(model.transformer[l].register_forward_hook(make_hook(l)))

    # Process windows
    n_samples = min(args.n_samples, len(windows))
    batch_size = 64
    with torch.no_grad():
        for i in range(0, n_samples, batch_size):
            batch = torch.from_numpy(np.stack(windows[i:i+batch_size])).float().to(device)
            s1_ids, s2_ids = tokenizer.encode(batch, half=True)
            model(s1_ids, s2_ids)

    for h in hooks: h.remove()

    for l in layers:
        if all_acts_raw[l]:
            acts[l] = np.concatenate(all_acts_raw[l], axis=0)
            print(f"  Layer {l}: {acts[l].shape}")

    # Interpret each layer
    results = {}
    for layer in layers:
        if layer not in acts:
            continue
        print(f"\n{'='*50}")
        print(f"Interpreting Layer {layer}")
        print(f"{'='*50}")

        # Load SAE
        sae = TopKSAE(d_model, d_hidden, k=64).to(device)
        sae_path = Path(args.sae_dir) / f"sae_layer{layer}.pt"
        if not sae_path.exists():
            print(f"  SKIP: no SAE at {sae_path}")
            continue
        sae.load_state_dict(torch.load(str(sae_path), map_location=device, weights_only=True))
        sae.eval()

        layer_results = interpret_layer(sae, acts[layer], labels, device)
        results[f"layer_{layer}"] = layer_results

        # Print summary
        if "type_distribution" in layer_results:
            print(f"\n  Feature type distribution:")
            for t, c in sorted(layer_results["type_distribution"].items(), key=lambda x: -x[1]):
                pct = 100 * c / layer_results["n_alive"]
                print(f"    {t}: {c} features ({pct:.1f}%)")

        if "clusters" in layer_results:
            print(f"\n  Concept clusters:")
            for cid, cinfo in sorted(layer_results["clusters"].items()):
                print(f"    Cluster {cid}: {cinfo['name']} ({cinfo['size']} features, r={cinfo['mean_correlation']:.3f})")

        if "feature_correlation_range" in layer_results:
            fcr = layer_results["feature_correlation_range"]
            print(f"\n  Correlation stats: max|r|={fcr['max_abs_corr']:.3f}, "
                  f"mean|r|={fcr['mean_abs_corr']:.3f}, "
                  f"n_high_corr={fcr['n_high_corr']}/{layer_results['n_alive']}")

        del sae
        torch.cuda.empty_cache()

    # Save
    with open(args.output, "w") as f:
        json.dump(results, f, indent=2, default=str)

    print(f"\n{'='*60}")
    print(f"Interpretability results saved to {args.output}")
    print(f"Done in {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
