import numpy as np, re
from scipy import stats

k_vols, c_vols, r_vols = [], [], []
with open("/data/houwanlong/finllm-mi/logs/scale.log") as f:
    for line in f:
        m = re.search(r"N=\d+, test=\d+, K=([\d.]+), C=([\d.]+), Raw=([\d.]+)", line)
        if m:
            k_vols.append(float(m.group(1)))
            c_vols.append(float(m.group(2)))
            r_vols.append(float(m.group(3)))

print(f"Stocks: {len(k_vols)}")

t_kc, p_kc = stats.ttest_rel(k_vols, c_vols)
t_kr, p_kr = stats.ttest_rel(k_vols, r_vols)
t_cr, p_cr = stats.ttest_rel(c_vols, r_vols)

d_kc = np.mean([k-c for k,c in zip(k_vols, c_vols)])
d_kr = np.mean([k-r for k,r in zip(k_vols, r_vols)])
d_cr = np.mean([c-r for c,r in zip(c_vols, r_vols)])

print(f"K vs C:   delta={d_kc:+.4f}, t={t_kc:.3f}, p={p_kc:.6f}  SIG={p_kc<0.05}")
print(f"K vs Raw: delta={d_kr:+.4f}, t={t_kr:.3f}, p={p_kr:.6f}  SIG={p_kr<0.05}")
print(f"C vs Raw: delta={d_cr:+.4f}, t={t_cr:.3f}, p={p_cr:.6f}  SIG={p_cr<0.05}")
print(f"K>C in {sum(1 for k,c in zip(k_vols,c_vols) if k>c)}/{len(k_vols)}")
print(f"K>Raw in {sum(1 for k,r in zip(k_vols,r_vols) if k>r)}/{len(k_vols)}")
print(f"C>Raw in {sum(1 for c,r in zip(c_vols,r_vols) if c>r)}/{len(k_vols)}")
