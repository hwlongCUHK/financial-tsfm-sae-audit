import numpy as np, glob, sys

for dp in [30, 40, 50]:
    files = sorted(glob.glob(f'/home/houwanlong/marketing_kg/asset_allocation/Signature-Informed-Transformer-For-Asset-Allocation-main/attention_data/dp{dp}_*_attention.npz'))
    all_a = []
    for f in files:
        d = np.load(f, allow_pickle=True)
        all_a.append(d['attentions'])
    all_a = np.concatenate(all_a, axis=0)
    attn_per_sample = all_a.mean(axis=(1,2,3))
    n = len(attn_per_sample)

    attn_mean = attn_per_sample.mean(axis=0)
    attn_std = attn_per_sample.std(axis=0)

    # Adjacent vs far correlation
    adj = np.corrcoef(attn_per_sample[:n-1].reshape(n-1,-1), attn_per_sample[1:].reshape(n-1,-1))
    adj_corr = adj[:n-1, n-1:].diagonal().mean()
    far = np.corrcoef(attn_per_sample[:n//2].reshape(n//2,-1), attn_per_sample[n//2:].reshape(n//2,-1))
    far_corr = far[:n//2, n//2:].diagonal().mean()

    eps = 1e-8
    a_clip = np.clip(attn_per_sample, eps, 1)
    a_clip = a_clip / a_clip.sum(axis=-1, keepdims=True)
    entropy = (-a_clip * np.log(a_clip)).sum(axis=(-1,-2)).mean()
    max_entropy = np.log(dp * dp)

    print(f"dp={dp}: {n} samples")
    print(f"  Mean self-attn: {attn_mean.diagonal().mean():.3f} (1/dp={1/dp:.3f})")
    print(f"  Std(attn) mean={attn_std.mean():.4f} Entropy={entropy:.2f}/{max_entropy:.2f}")
    print(f"  Adjacent corr: {adj_corr:.3f}, Far corr: {far_corr:.3f}")
    print()
