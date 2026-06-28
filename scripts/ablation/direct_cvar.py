"""Direct signature-to-weights with CVaR training. No diffusion. Tests the core hypothesis."""
import torch, torch.nn as nn, numpy as np, sys
sys.path.insert(0, "/home/houwanlong/marketing_kg/asset_allocation/Signature-Informed-Transformer-For-Asset-Allocation-main")
from data_provider.data_loader import PrecomputedSigDataset

class DirectSig2Weight(nn.Module):
    def __init__(self, N):
        super().__init__()
        self.N = N
        self.net = nn.Sequential(
            nn.Linear(N*N, 256), nn.ReLU(),
            nn.Linear(256, 128), nn.ReLU(),
            nn.Linear(128, N)
        )
    def forward(self, sig):
        B = sig.shape[0]
        logits = self.net(sig.reshape(B, -1))
        return torch.softmax(logits / 1.3, dim=-1)

for N in [30, 40, 50]:
    print("dp=%d" % N)
    ds_train = PrecomputedSigDataset("./signature_cache_6020/pool_%d" % N, "train")
    ds_test = PrecomputedSigDataset("./signature_cache_6020/pool_%d" % N, "test")

    sigs, rets = [], []
    for i in range(min(2000, len(ds_train))):
        s = ds_train[i]
        sigs.append(torch.FloatTensor(s["cross_sigs"].mean(axis=0).squeeze(-1)))
        rets.append(torch.FloatTensor(s["future_return_unscaled"].mean(axis=0)))
    sigs = torch.stack(sigs); rets = torch.stack(rets)
    sm, ss = sigs.mean(), sigs.std()
    sigs = (sigs - sm) / (ss + 1e-8)
    device = torch.device("cuda")

    model = DirectSig2Weight(N).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)

    for epoch in range(200):
        perm = torch.randperm(len(sigs)); total_loss = 0; nb = 0
        for start in range(0, len(sigs), 64):
            idx = perm[start:start+64]; s = sigs[idx].to(device); r = rets[idx].to(device)
            w = model(s); pr = (w * r).sum(dim=-1)
            k = max(1, int(len(idx) * 0.05)); loss = -pr.sort()[0][:k].mean()
            opt.zero_grad(); loss.backward(); opt.step()
            total_loss += loss.item(); nb += 1
        if epoch % 50 == 0:
            print("  epoch %d: loss=%.4f" % (epoch, total_loss/nb))

    model.eval()
    test_sigs, test_rets = [], []
    for i in range(0, len(ds_test), 5):
        s = ds_test[i]
        test_sigs.append(torch.FloatTensor(s["cross_sigs"].mean(axis=0).squeeze(-1)))
        test_rets.append(torch.FloatTensor(s["future_return_unscaled"].mean(axis=0)))
    test_sigs = torch.stack(test_sigs); test_rets = torch.stack(test_rets)
    test_sigs = (test_sigs - sm) / (ss + 1e-8)

    with torch.no_grad():
        w = model(test_sigs.to(device)).cpu().numpy()
    rets_np = test_rets.numpy()
    ew = np.ones(N)/N
    sr_ew = (rets_np*ew).sum(axis=1); sr_ew = sr_ew.mean()/sr_ew.std()*np.sqrt(252)
    sr_m = (rets_np*w).sum(axis=1); sr_m = sr_m.mean()/sr_m.std()*np.sqrt(252)
    print("  EW=%.4f  DirectCVaR=%.4f  diff=%+.4f" % (sr_ew, sr_m, sr_m-sr_ew))
