"""LLM auto-interpretability: use DeepSeek to label SAE features with financial concepts."""
import torch, numpy as np, json, sys, time
from pathlib import Path
sys.path.insert(0, "/data/houwanlong/finllm-mi/code")
from model.kronos import Kronos, KronosTokenizer
from safetensors.torch import load_file
import pandas as pd
from openai import OpenAI

API_KEY = "sk-ca6f57ac52fd4d4cbbb12d583ff68f82"
LAYERS = [0, 3, 6, 9, 11]
DEVICE = "cuda:0"
SAE_DIR = "/data/houwanlong/finllm-mi/outputs/sae"
N_FEATURES_PER_LAYER = 30  # Top 30 features per layer

class TopKSAE(torch.nn.Module):
    def __init__(self, d_model, d_hidden, k=64):
        super().__init__()
        self.encoder = torch.nn.Linear(d_model, d_hidden, bias=True)
        self.decoder = torch.nn.Linear(d_hidden, d_model, bias=False)
        self.b_pre = torch.nn.Parameter(torch.zeros(d_model))
        self.k = k
    def forward(self, x):
        xc = x - self.b_pre; lat = self.encoder(xc)
        _, idx = torch.topk(lat, self.k, dim=-1)
        mask = torch.zeros_like(lat); mask.scatter_(-1, idx, 1.0)
        return self.decoder(lat * mask) + self.b_pre, lat * mask

print("Loading Kronos...")
tokenizer = KronosTokenizer.from_pretrained("/data/houwanlong/models/Kronos-Tokenizer-base").to(DEVICE).eval()
with open("/data/houwanlong/models/Kronos-base/config.json") as f: cfg = json.load(f)
model = Kronos(s1_bits=cfg["s1_bits"], s2_bits=cfg["s2_bits"], n_layers=cfg["n_layers"],
               d_model=cfg["d_model"], n_heads=cfg["n_heads"], ff_dim=cfg["ff_dim"],
               ffn_dropout_p=cfg["ffn_dropout_p"], attn_dropout_p=cfg["attn_dropout_p"],
               resid_dropout_p=cfg["resid_dropout_p"], token_dropout_p=cfg["token_dropout_p"],
               learn_te=cfg["learn_te"])
sd = load_file("/data/houwanlong/models/Kronos-base/model.safetensors")
model.load_state_dict(sd, strict=False); model = model.to(DEVICE).half().eval()

# Load Alibaba data for interpretability
df = pd.read_csv("/data/houwanlong/finllm-mi/code/finetune_csv/data/HK_ali_09988_kline_5min_all.csv")
for col in ["open","close","high","low","volume","amount"]:
    if col not in df.columns: df[col] = 0.0
data = df[["open","close","high","low","volume","amount"]].values.astype(np.float32)
data = data[~np.isnan(data).any(axis=1)]
# Keep raw prices for interpretability
raw_data = data.copy()
mn, st = data.mean(0), data.std(0)
data_norm = np.clip((data - mn) / (st + 1e-5), -5, 5)

lookback, stride = 64, 32
n_train = min(2000, (len(data_norm) - lookback) // stride)
windows = np.stack([data_norm[i:i+lookback] for i in range(0, n_train * stride, stride)])
raw_windows = np.stack([raw_data[i:i+lookback] for i in range(0, n_train * stride, stride)])

client = OpenAI(api_key=API_KEY, base_url="https://api.deepseek.com")

results = {}

for layer in LAYERS:
    print(f"\n{chr(61)*50}")
    print(f"Layer {layer}")
    print(f"{chr(61)*50}")

    # Load SAE
    sae_path = Path(SAE_DIR) / f"sae_layer{layer}.pt"
    if not sae_path.exists():
        print(f"  No SAE found")
        continue
    sae = TopKSAE(832, 832*4).to(DEVICE)
    sae.load_state_dict(torch.load(str(sae_path), map_location=DEVICE, weights_only=True))
    sae.eval()

    # Collect activations
    acts_list = []
    def hook_fn(m, i, o):
        a = o[0] if isinstance(o, tuple) else o
        acts_list.append(a[:, -1, :].detach().cpu().float().numpy())
    hook = model.transformer[layer].register_forward_hook(hook_fn)
    bs = 64
    with torch.no_grad():
        for b in range(0, len(windows), bs):
            batch = torch.from_numpy(windows[b:b+bs]).float().to(DEVICE)
            s1, s2 = tokenizer.encode(batch, half=True)
            model(s1, s2)
    hook.remove()
    acts = np.concatenate(acts_list, axis=0)

    # Get feature activations
    acts_t = torch.from_numpy(acts).float().to(DEVICE)
    with torch.no_grad():
        _, latents = sae(acts_t)
    latents_np = latents.cpu().numpy()

    # Find top features by activation frequency
    feat_freq = (latents_np != 0).sum(axis=0)
    top_features = feat_freq.argsort()[::-1][:N_FEATURES_PER_LAYER]

    # For each top feature, get top-5 activating windows
    layer_labels = []
    for feat_idx in top_features:
        feat_acts = latents_np[:, feat_idx]
        top_wins = feat_acts.argsort()[::-1][:5]

        # Build prompt with activating window statistics
        examples = []
        for i, wi in enumerate(top_wins):
            win = raw_windows[wi]  # (64, 6) raw prices
            close = win[:, 1]
            rets = np.diff(close) / (close[:-1] + 1e-5)
            examples.append(
                f"Example {i+1}: close={close[0]:.2f}→{close[-1]:.2f}, "
                f"volatility={np.std(rets):.4f}, trend={'UP' if close[-1]>close[0] else 'DOWN'}, "
                f"range={close.max()-close.min():.2f}, "
                f"max_dd={np.min(close/np.maximum.accumulate(close)-1):.3f}"
            )

        prompt = f"""You are analyzing features discovered by a sparse autoencoder inside Kronos, a financial time series foundation model.

Here are the top 5 K-line windows that maximally activate Feature #{feat_idx} in Layer {layer}:

{chr(10).join(examples)}

Based on these activating windows, what financial concept does this feature most likely represent?
Respond with ONLY a short label (2-4 words), choosing from or similar to:
- Volatility Spike Detector
- Trend Follower (Up)
- Trend Follower (Down)
- Crash / Tail Risk Detector
- Mean Reversion Detector
- Price Range / Consolidation
- Volume Surge Detector
- Volatility Clustering / Regime Shift
- Momentum Detector
- Gap Detector
- Sideways / Low Vol
- Other (specify briefly)

Just give the label, nothing else."""

        try:
            resp = client.chat.completions.create(
                model="deepseek-chat",
                messages=[{"role": "user", "content": prompt}],
                max_tokens=20,
                temperature=0.1,
            )
            label = resp.choices[0].message.content.strip()
        except Exception as e:
            label = f"API_ERROR: {str(e)[:50]}"

        layer_labels.append({
            "feature_id": int(feat_idx),
            "label": label,
            "activation_count": int(feat_freq[feat_idx]),
        })
        if len(layer_labels) % 10 == 0:
            print(f"  Labeled {len(layer_labels)}/{N_FEATURES_PER_LAYER} features...")

    # Aggregate labels
    from collections import Counter
    label_counts = Counter(l["label"] for l in layer_labels)
    results[f"layer_{layer}"] = {
        "features": layer_labels,
        "label_distribution": dict(label_counts.most_common(10)),
    }

    print(f"  Top labels: {label_counts.most_common(5)}")
    del sae; torch.cuda.empty_cache()
    time.sleep(0.5)  # Rate limit

# Save
with open(f"{SAE_DIR}/llm_labels.json", "w") as f:
    json.dump(results, f, indent=2)
print(f"\nSaved to {SAE_DIR}/llm_labels.json")
