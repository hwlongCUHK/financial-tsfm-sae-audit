"""Fast fine-tune Chronos-T5-Mini on 80 stocks, SAE before/after on 20 held-out."""
import os, json, time, numpy as np, pandas as pd, torch, sys, warnings
warnings.filterwarnings("ignore")
from pathlib import Path
from transformers import T5ForConditionalGeneration, get_linear_schedule_with_warmup

device = "cuda:0"
DATA = Path("/data/houwanlong/finllm-mi/data/scale120")
OUTPUT = "/data/houwanlong/finllm-mi/outputs/sae/chronos_refinetune.json"
MODEL_OUT = "/data/houwanlong/models/chronos-t5-small-fin/"

np.random.seed(42); torch.manual_seed(42)

# Split stocks
all_files = sorted([f.name for f in DATA.glob("*.csv")])
np.random.shuffle(all_files)
train_files = all_files[:80]; test_files = all_files[80:100]
print(f"Train: {len(train_files)}, Test: {len(test_files)}")

# Load model
print("Loading Chronos-T5-Mini...")
model = T5ForConditionalGeneration.from_pretrained("/data/houwanlong/models/chronos-t5-small/", torch_dtype=torch.float16).to(device)
model.train()

# Load training data
LOOKBACK, PRED_LEN = 64, 16
all_inputs, all_targets = [], []
for fname in train_files[:30]:  # Use 30 stocks for speed
    df = pd.read_csv(DATA / fname)
    if "close" not in df.columns: continue
    close = df["close"].values.astype(np.float32)
    close = close[~np.isnan(close)]
    if len(close) < LOOKBACK + PRED_LEN + 10: continue
    # Normalize
    mn, st = close.mean(), close.std()
    close_norm = (close - mn) / (st + 1e-5)
    for i in range(0, len(close_norm) - LOOKBACK - PRED_LEN, 32):
        inp = close_norm[i:i+LOOKBACK]
        tgt = close_norm[i+LOOKBACK:i+LOOKBACK+PRED_LEN]
        all_inputs.append(inp); all_targets.append(tgt)

print(f"Training samples: {len(all_inputs)}")
if len(all_inputs) < 100:
    print("Not enough data!"); sys.exit(0)

# Tokenize (simple: bin into 4096 levels for Chronos)
def tokenize(x): return ((x - x.min()) / (x.max() - x.min() + 1e-5) * 4095).astype(np.int64)

batch_size = 8
opt = torch.optim.AdamW(model.parameters(), lr=1e-4)
steps = min(500, len(all_inputs) // batch_size)

for step in range(steps):
    idx = np.random.choice(len(all_inputs), batch_size, replace=False)
    batch_inp = torch.from_numpy(np.stack([tokenize(all_inputs[i]) for i in idx])).long().to(device)
    batch_tgt = torch.from_numpy(np.stack([tokenize(all_targets[i]) for i in idx])).long().to(device)
    loss = model(input_ids=batch_inp, labels=batch_tgt).loss
    opt.zero_grad(); loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step()
    if step % 100 == 0: print(f"  Step {step}: loss={loss.item():.4f}")

# Save
os.makedirs(MODEL_OUT, exist_ok=True)
model.save_pretrained(MODEL_OUT)
print(f"Saved to {MODEL_OUT}")

# Quick SAE comparison on 10 test stocks
from scipy import stats

class SAE(torch.nn.Module):
    def __init__(self, d, h, k=64):
        super().__init__()
        self.enc = torch.nn.Linear(d, h, bias=True)
        self.dec = torch.nn.Linear(h, d, bias=False)
        self.b = torch.nn.Parameter(torch.zeros(d))
        self.k = k
    def encode(self, x):
        xc = x - self.b
        lat = self.enc(xc)
        _, idx = torch.topk(lat, self.k, dim=-1)
        m = torch.zeros_like(lat)
        m.scatter_(-1, idx, 1.0)
        return lat * m
    def decode(self, lat):
        return self.dec(lat) + self.b
