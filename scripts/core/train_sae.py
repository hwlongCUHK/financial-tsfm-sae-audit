"""Train TopK SAEs on Kronos layers — following TimeSAE/Chronos SAE methodology."""
import torch
import torch.nn as nn
import numpy as np
import json, sys, time, argparse
from pathlib import Path
from safetensors.torch import load_file

sys.path.insert(0, "/data/houwanlong/finllm-mi/code")
from model.kronos import Kronos, KronosTokenizer


class TopKSAE(nn.Module):
    """TopK Sparse Autoencoder following Bricken et al. / Chronos SAE."""
    def __init__(self, d_model, d_hidden, k=32, tied_weights=False):
        super().__init__()
        self.d_model = d_model
        self.d_hidden = d_hidden
        self.k = k

        self.encoder = nn.Linear(d_model, d_hidden, bias=True)
        self.decoder = nn.Linear(d_hidden, d_model, bias=False)
        if tied_weights:
            self.decoder.weight = nn.Parameter(self.encoder.weight.T.clone())

        # Pre-encoder bias (Bricken et al.)
        self.b_pre = nn.Parameter(torch.zeros(d_model))

    def forward(self, x):
        # Subtract pre-encoder bias
        x_centered = x - self.b_pre

        # Encode
        latents = self.encoder(x_centered)

        # TopK activation
        topk_vals, topk_idx = torch.topk(latents, self.k, dim=-1)
        mask = torch.zeros_like(latents)
        mask.scatter_(-1, topk_idx, 1.0)
        latents = latents * mask

        # Decode
        x_recon = self.decoder(latents) + self.b_pre

        # Loss
        l_recon = nn.functional.mse_loss(x_recon, x)
        l_aux = self._dead_feature_aux_loss(x_centered, latents)

        # L0 sparsity (for logging)
        l0 = (latents != 0).float().sum(dim=-1).mean()

        return x_recon, latents, l_recon, l_aux, l0

    def _dead_feature_aux_loss(self, x, latents):
        """Auxiliary loss to revive dead features."""
        # Count feature usage
        alive = (latents != 0).float().sum(dim=0) > 0
        dead_frac = 1.0 - alive.float().mean()

        if dead_frac < 0.01:
            return torch.tensor(0.0, device=x.device)

        # For dead features, try to reconstruct using them
        dead_idx = torch.where(~alive)[0]
        if len(dead_idx) == 0:
            return torch.tensor(0.0, device=x.device)

        # Reconstruct residual for dead features
        x_recon_dead = self.decoder(latents)
        residual = x - x_recon_dead
        dead_latent = self.encoder(residual - self.b_pre)
        dead_latent_masked = torch.zeros_like(dead_latent)
        dead_latent_masked[:, dead_idx] = dead_latent[:, dead_idx]
        x_dead_recon = self.decoder(dead_latent_masked) + self.b_pre
        aux_loss = nn.functional.mse_loss(x_dead_recon, residual)
        return aux_loss


def collect_kronos_activations(tokenizer, model, data_path, n_samples, lookback, device):
    """Collect residual stream activations from Kronos."""
    import pandas as pd

    df = pd.read_csv(data_path)
    for col in ["open","close","high","low","volume","amount"]:
        if col not in df.columns: df[col] = 0.0
    data = df[["open","close","high","low","volume","amount"]].values.astype(np.float32)
    data = data[~np.isnan(data).any(axis=1)]
    mn, st = data.mean(0), data.std(0)
    data_norm = np.clip((data - mn) / (st + 1e-5), -5, 5)

    stride = lookback // 2  # 50% overlap for more samples
    n_windows = min(n_samples, (len(data_norm) - lookback) // stride)

    windows = []
    for i in range(0, n_windows * stride, stride):
        windows.append(data_norm[i:i+lookback])

    windows = np.stack(windows, axis=0)
    print(f"  Windows: {windows.shape}")

    # Hook all layers
    n_layers = len(model.transformer)
    acts = {l: [] for l in range(n_layers)}

    def make_hook(layer_idx):
        def h(module, input, output):
            act = output[0] if isinstance(output, tuple) else output
            # Save last-token residual stream
            acts[layer_idx].append(act[:, -1, :].detach().cpu().float().numpy())
        return h

    hooks = [model.transformer[l].register_forward_hook(make_hook(l)) for l in range(n_layers)]

    batch_size = 64
    with torch.no_grad():
        for b in range(0, len(windows), batch_size):
            batch = torch.from_numpy(windows[b:b+batch_size]).float().to(device)
            s1_ids, s2_ids = tokenizer.encode(batch, half=True)
            model(s1_ids, s2_ids)

    for h in hooks: h.remove()

    return {l: np.concatenate(acts[l], axis=0) for l in range(n_layers)}


def train_sae(acts, d_hidden, k, lr, steps, batch_size, device, layer_name):
    """Train a single SAE on one layer's activations."""
    d_model = acts.shape[1]
    sae = TopKSAE(d_model, d_hidden, k=k).to(device)
    optimizer = torch.optim.Adam(sae.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, steps)

    acts_tensor = torch.from_numpy(acts).float().to(device)
    n_samples = acts_tensor.shape[0]

    best_loss = float('inf')
    log = []

    for step in range(steps):
        idx = torch.randint(0, n_samples, (batch_size,))
        x = acts_tensor[idx]

        x_recon, latents, l_recon, l_aux, l0 = sae(x)
        loss = l_recon + 0.01 * l_aux

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(sae.parameters(), 1.0)
        optimizer.step()
        scheduler.step()

        if step % 200 == 0 or step == steps - 1:
            # Compute dead feature fraction
            with torch.no_grad():
                full_latents = sae.encoder(acts_tensor[:min(2000, n_samples)] - sae.b_pre)
                topk_vals, _ = torch.topk(full_latents, k, dim=-1)
                alive = (full_latents.abs().sum(dim=0) > 1e-6).float().mean()

            info = {
                "step": step,
                "loss": float(loss.item()),
                "l_recon": float(l_recon.item()),
                "l_aux": float(l_aux.item()),
                "l0": float(l0.item()),
                "alive_frac": float(alive.item()),
                "lr": float(scheduler.get_last_lr()[0]),
            }
            log.append(info)
            if step % 1000 == 0 or step == steps - 1:
                print(f"  {layer_name} step {step}: recon={l_recon.item():.4f} aux={l_aux.item():.4f} "
                      f"L0={l0.item():.1f} alive={alive.item():.1%}")

            if l_recon.item() < best_loss:
                best_loss = l_recon.item()
                best_state = {k: v.cpu().clone() for k, v in sae.state_dict().items()}

    # Load best state
    sae.load_state_dict(best_state)
    return sae, log


def evaluate_sae(sae, acts, device):
    """Evaluate SAE: reconstruction quality and feature interpretability score."""
    acts_tensor = torch.from_numpy(acts[:2000]).float().to(device)
    with torch.no_grad():
        x_recon, latents, l_recon, _, l0 = sae(acts_tensor)
        # Variance explained
        var_explained = 1 - l_recon.item() / acts_tensor.var().item()
        # Top activating samples per feature
        top_acts = []
        for feat in range(min(20, sae.d_hidden)):
            feat_acts = latents[:, feat]
            top_idx = feat_acts.argsort(descending=True)[:5]
            top_acts.append(top_idx.cpu().numpy().tolist())

    return {
        "var_explained": float(var_explained),
        "l_recon": float(l_recon.item()),
        "l0": float(l0.item()),
        "alive_features": float((latents.abs().sum(dim=0) > 1e-6).float().mean().item()),
        "top_activating_examples": top_acts,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="/data/houwanlong/finllm-mi/code/finetune_csv/data/HK_ali_09988_kline_5min_all.csv")
    parser.add_argument("--output", default="/data/houwanlong/finllm-mi/outputs/sae")
    parser.add_argument("--n-samples", type=int, default=100000)
    parser.add_argument("--lookback", type=int, default=64)
    parser.add_argument("--expansion", type=int, default=8)
    parser.add_argument("--k", type=int, default=32)
    parser.add_argument("--steps", type=int, default=5000)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--layers", type=str, default="all")
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()

    device = args.device
    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)

    t0 = time.time()

    # Load models
    print("Loading Kronos...")
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

    # Collect activations
    print(f"\nCollecting {args.n_samples} activation samples...")
    all_acts = collect_kronos_activations(tokenizer, model, args.data, args.n_samples, args.lookback, device)
    print(f"Collected activations for {len(all_acts)} layers")

    # Determine layers to train
    if args.layers == "all":
        layers = list(all_acts.keys())
    else:
        layers = [int(x) for x in args.layers.split(",")]

    d_model = all_acts[0].shape[1]
    d_hidden = d_model * args.expansion
    print(f"\nSAE config: d_model={d_model}, d_hidden={d_hidden}, k={args.k}, expansion={args.expansion}x")
    print(f"Training on layers: {layers}")
    print(f"Steps: {args.steps}, LR: {args.lr}, Batch: {args.batch_size}\n")

    results = {}
    for layer in layers:
        print(f"\n{'='*50}")
        print(f"Training SAE on Layer {layer}")
        print(f"{'='*50}")

        sae, log = train_sae(
            all_acts[layer], d_hidden, args.k, args.lr, args.steps, args.batch_size, device,
            f"L{layer}"
        )

        # Evaluate
        evals = evaluate_sae(sae, all_acts[layer], device)
        print(f"  Var explained: {evals['var_explained']:.4f}, Alive: {evals['alive_features']:.1%}")

        # Save
        torch.save(sae.state_dict(), str(out_dir / f"sae_layer{layer}.pt"))
        results[f"layer_{layer}"] = {"eval": evals, "train_log": log}

    # Save results
    with open(str(out_dir / "sae_results.json"), "w") as f:
        json.dump(results, f, indent=2, default=str)

    print(f"\nDone in {time.time() - t0:.0f}s. Results saved to {out_dir}")


if __name__ == "__main__":
    main()
