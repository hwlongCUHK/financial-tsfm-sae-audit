"""Causal validation: SAE feature ablation → measure impact on Kronos forecasting."""
import torch
import numpy as np
import json, sys, time, argparse
from pathlib import Path
import pandas as pd
from scipy import stats

sys.path.insert(0, "/data/houwanlong/finllm-mi/code")
from model.kronos import Kronos, KronosTokenizer
from safetensors.torch import load_file


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
    def reconstruct_with_ablation(self, x, ablate_features):
        """Reconstruct with specific features ablated (set to 0)."""
        x_centered = x - self.b_pre
        latents = self.encoder(x_centered)
        topk_vals, topk_idx = torch.topk(latents, self.k, dim=-1)
        mask = torch.zeros_like(latents)
        mask.scatter_(-1, topk_idx, 1.0)
        # Ablate specified features
        mask[:, ablate_features] = 0
        latents = latents * mask
        return self.decoder(latents) + self.b_pre


def load_kronos_full(device):
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
    return tokenizer, model


def compute_forecast_quality(model_acts_original, model_acts_ablated):
    """Measure how much ablating SAE features changes model output."""
    # Cosine similarity between original and ablated residual streams
    cos_sim = torch.nn.functional.cosine_similarity(
        model_acts_original.float(), model_acts_ablated.float(), dim=-1
    )
    return cos_sim


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sae-dir", default="/data/houwanlong/finllm-mi/outputs/sae")
    parser.add_argument("--output", default="/data/houwanlong/finllm-mi/outputs/sae/causal_validation.json")
    parser.add_argument("--layers", type=str, default="3,6,9")
    parser.add_argument("--n-samples", type=int, default=200)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()

    device = args.device
    layers = [int(x) for x in args.layers.split(",")]
    d_model = 832
    d_hidden = d_model * 4
    t0 = time.time()

    print("Loading models...")
    tokenizer, model = load_kronos_full(device)

    # Load data
    df = pd.read_csv("/data/houwanlong/finllm-mi/code/finetune_csv/data/HK_ali_09988_kline_5min_all.csv")
    for col in ["open","close","high","low","volume","amount"]:
        if col not in df.columns: df[col] = 0.0
    data = df[["open","close","high","low","volume","amount"]].values.astype(np.float32)
    data = data[~np.isnan(data).any(axis=1)]
    mn, st = data.mean(0), data.std(0)
    data_norm = np.clip((data - mn) / (st + 1e-5), -5, 5)

    # Create test windows
    lookback = 64
    windows = []
    for i in range(0, min(args.n_samples * 64, len(data_norm) - lookback), 64):
        windows.append(data_norm[i:i+lookback])
    windows = np.stack(windows[:args.n_samples])

    results = {}
    for layer in layers:
        print(f"\n{'='*60}")
        print(f"Causal Validation: Layer {layer}")
        print(f"{'='*60}")

        # Load SAE
        sae = TopKSAE(d_model, d_hidden, k=64).to(device)
        sae_path = Path(args.sae_dir) / f"sae_layer{layer}.pt"
        sae.load_state_dict(torch.load(str(sae_path), map_location=device, weights_only=True))
        sae.eval()

        # Load interpretability results
        interp_path = Path(args.sae_dir) / "interpretability.json"
        with open(interp_path) as f:
            interp = json.load(f)
        layer_interp = interp.get(f"layer_{layer}", {})
        type_dist = layer_interp.get("type_distribution", {})

        # Register hook to intercept residual stream
        original_acts = []
        ablated_acts = {}

        def make_hook():
            def h(module, input, output):
                if isinstance(output, tuple):
                    original_acts.append(output[0].detach())
                else:
                    original_acts.append(output.detach())
            return h

        hook = model.transformer[layer].register_forward_hook(make_hook())

        # Process one batch to get original activations
        batch = torch.from_numpy(windows[:min(32, args.n_samples)]).float().to(device)
        with torch.no_grad():
            s1_ids, s2_ids = tokenizer.encode(batch, half=True)
            _ = model(s1_ids, s2_ids)

        hook.remove()
        orig_act = original_acts[0]  # (batch, seq, d_model)
        # Take last token residual
        orig_residual = orig_act[:, -1, :]  # (batch, d_model)

        # Get SAE features and their activations
        with torch.no_grad():
            _, latents = sae(orig_residual)

        # Find top-activating features
        feat_usage = (latents != 0).float().sum(dim=0)  # per-feature activation count
        top_features = feat_usage.argsort(descending=True)[:100]  # top 100 most active

        # === EXPERIMENT 1: Ablate top features, measure reconstruction change ===
        print("\nExperiment 1: Feature Ablation → Reconstruction Impact")
        with torch.no_grad():
            full_recon, _ = sae(orig_residual)

            # Ablate top 10, 20, 50, 100 features
            ablations = []
            for n_ablate in [10, 20, 50, 100]:
                ablate_ids = top_features[:n_ablate].tolist()
                recon_ablated = sae.reconstruct_with_ablation(orig_residual, ablate_ids)
                cos_sim = torch.nn.functional.cosine_similarity(
                    full_recon.float(), recon_ablated.float(), dim=-1).mean()
                mse_increase = torch.nn.functional.mse_loss(recon_ablated, orig_residual).item()
                base_mse = torch.nn.functional.mse_loss(full_recon, orig_residual).item()
                rel_increase = (mse_increase - base_mse) / (base_mse + 1e-10)
                ablations.append({
                    "n_ablated": n_ablate,
                    "cos_sim": float(cos_sim.item()),
                    "mse_rel_increase": float(rel_increase),
                })
                print(f"  Ablate top {n_ablate}: cos_sim={cos_sim.item():.4f}, MSE increase={rel_increase:.1%}")

        # === EXPERIMENT 2: Ablate financial-concept features ===
        print("\nExperiment 2: Concept-Specific Ablation")
        # Get feature-concept mapping from interpretability results
        top_features_data = layer_interp.get("top_features_by_type", {})

        concept_ablations = {}
        for concept_name, feat_list in top_features_data.items():
            if not feat_list:
                continue
            concept_ids = [f["feature_id"] for f in feat_list[:5]]  # top 5 per concept
            with torch.no_grad():
                recon_ablated = sae.reconstruct_with_ablation(orig_residual, concept_ids)
                cos_sim = torch.nn.functional.cosine_similarity(
                    full_recon.float(), recon_ablated.float(), dim=-1).mean()
                mse_inc = torch.nn.functional.mse_loss(recon_ablated, orig_residual).item()
                base_mse = torch.nn.functional.mse_loss(full_recon, orig_residual).item()
                concept_ablations[concept_name] = {
                    "n_features": len(concept_ids),
                    "cos_sim": float(cos_sim.item()),
                    "mse_rel_increase": float((mse_inc - base_mse) / (base_mse + 1e-10)),
                }
                print(f"  Ablate {concept_name} (top 5): cos_sim={cos_sim.item():.4f}")

        # === EXPERIMENT 3: Random ablation baseline ===
        print("\nExperiment 3: Random Ablation Baseline")
        random_results = []
        for _ in range(10):
            random_ids = np.random.choice(d_hidden, 20, replace=False).tolist()
            with torch.no_grad():
                recon_rand = sae.reconstruct_with_ablation(orig_residual, random_ids)
                cos_sim = torch.nn.functional.cosine_similarity(
                    full_recon.float(), recon_rand.float(), dim=-1).mean()
                random_results.append(float(cos_sim.item()))

        random_mean = np.mean(random_results)
        random_std = np.std(random_results)
        print(f"  Random 20 features: cos_sim={random_mean:.4f} ± {random_std:.4f}")

        # Significance: top 20 real vs random 20
        top20_cos = ablations[1]["cos_sim"]  # from n_ablate=20
        z_score = (top20_cos - random_mean) / (random_std + 1e-10)
        p_val = 2 * stats.norm.sf(abs(z_score))
        print(f"  Top 20 vs Random: z={z_score:.2f}, p={p_val:.4f} {'SIG' if p_val < 0.05 else 'not sig'}")

        results[f"layer_{layer}"] = {
            "ablations": ablations,
            "concept_ablations": concept_ablations,
            "random_baseline": {"mean_cos_sim": random_mean, "std_cos_sim": random_std},
            "top20_vs_random": {"z_score": float(z_score), "p_value": float(p_val)},
        }

        del sae
        torch.cuda.empty_cache()

    with open(args.output, "w") as f:
        json.dump(results, f, indent=2)

    print(f"\n{'='*60}")
    print(f"Causal validation saved to {args.output}")
    print(f"Done in {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
