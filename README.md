# Interpreting Financial Time-Series Foundation Models with Sparse Autoencoders

Code repository for the ICAIF paper: *Interpreting Financial Time-Series Foundation Models with Sparse Autoencoders: An Empirical Audit of Kronos*.

## Abstract

We apply sparse autoencoders (SAEs) to audit the internal representations of Kronos, a pretrained financial time-series transformer. Training TopK SAEs on layer-6 residual stream activations from 120 Chinese A-share stocks, we find that Kronos encodes a broad ecology of basic financial signals -- momentum, volatility, autocorrelation, tail risk, price structure, and volume -- with no single signal dominating (largest concept family < 9% of assignments across 16 families). Intervention experiments (ablation and permutation) confirm that concept-labeled features carry structured financial information beyond activation frequency. Ablation increases prediction volatility by 23% while preserving directional structure, and the concept distribution is robust to SAE hyperparameters, random seeds, and layer choice.

## Project Structure

```
code/
├── README.md                           # This file
├── requirements.txt                    # Python dependencies
├── configs/
│   └── default.yaml                    # All experiment hyperparameters
├── scripts/
│   ├── 01_extract_activations.py       # Extract Kronos layer-6 activations
│   ├── 02_train_shared_sae.py          # Train shared SAE on 120 stocks
│   ├── 03_concept_labeling.py          # Statistics-first feature labeling
│   ├── 04_ablation_intervention.py     # Ablation and permutation tests
│   ├── 05_financial_validation.py      # Financial metric validation
│   ├── 06_sensitivity_analysis.py      # k/expansion sensitivity
│   ├── 07_visualize.py                 # Generate all figures
│   └── run_all.sh                      # Run full pipeline
├── src/
│   ├── __init__.py
│   ├── sae.py                          # TopK SAE class
│   ├── statistics.py                   # 30+ financial statistics computation
│   └── utils.py                        # Data loading, normalization utilities
└── figures/                            # Output figures (generated)
```

## Data Requirements

1. **Kronos model**: Kronos-base weights and tokenizer. Expected paths (configurable in `configs/default.yaml`):
   - `models/Kronos-base/config.json`
   - `models/Kronos-base/model.safetensors`
   - `models/Kronos-Tokenizer-base/`

2. **Stock data**: 120 Chinese A-share stock CSVs in `data/scale120/`. Each CSV must contain columns: `open`, `close`, `high`, `low`, `volume`, `amount`. The dataset covers four sectors (Bank, Energy, Technology, Consumer, 30 stocks each) from the Shanghai and Shenzhen exchanges (2005-2022).

3. **Kronos source code**: The Kronos model code must be importable. Add the Kronos code directory to `sys.path` or install it. The scripts expect `from model.kronos import Kronos, KronosTokenizer`.

## Setup

```bash
# Install dependencies
pip install -r requirements.txt

# Configure paths in configs/default.yaml:
#   - model.config_path, model.weights_path, model.tokenizer_path
#   - data.root
#   - device (e.g., "cuda:0")
```

## Reproducing Paper Results

### Full pipeline

```bash
bash scripts/run_all.sh
```

### Individual steps

Each script is standalone and can be run independently (provided its dependencies have been generated):

```bash
# Step 1: Extract activations from Kronos layer 6
python scripts/01_extract_activations.py

# Step 2: Train shared SAE on pooled activations
python scripts/02_train_shared_sae.py

# Step 3: Label SAE features with financial statistics
#         -> Produces Table 1 (concept distribution)
python scripts/03_concept_labeling.py

# Step 4: Ablation and permutation intervention tests
#         -> Produces Table 2 (dose-response), permutation results
python scripts/04_ablation_intervention.py

# Step 5: Financial metric validation
#         -> Produces Table 3 (financial validation metrics)
python scripts/05_financial_validation.py

# Step 6: Sensitivity analysis (k and expansion factor)
#         -> Produces Table 4 (sensitivity comparison)
python scripts/06_sensitivity_analysis.py

# Step 7: Generate all figures
python scripts/07_visualize.py
```

### Mapping scripts to paper tables and figures

| Paper Element | Script | Output File |
|---|---|---|
| Table 1 (Concept Distribution) | `03_concept_labeling.py` | `outputs/results/concept_labeling.json` |
| Table 2 (Ablation Dose-Response) | `04_ablation_intervention.py` | `outputs/results/ablation_results.json` |
| Table 3 (Financial Validation) | `05_financial_validation.py` | `outputs/results/financial_validation.json` |
| Table 4 (Sensitivity Analysis) | `06_sensitivity_analysis.py` | `outputs/results/sensitivity_results.json` |
| Figure 2 (Dose-Response Curve) | `07_visualize.py` | `figures/ablation_dose_response.pdf` |
| Section 3 (SAE Training Metrics) | `02_train_shared_sae.py` | `outputs/sae/shared_sae_results.json` |

## Key Hyperparameters

All hyperparameters are defined in `configs/default.yaml`:

| Parameter | Value | Description |
|---|---|---|
| Layer | 6 | Residual stream extraction layer |
| Window length | 64 | K-line periods per window |
| Stride | 32 | Sliding window stride |
| SAE k | 64 | Active features per sample |
| SAE expansion | 4x | d_hidden = 832 * 4 = 3328 |
| Training steps (per-stock) | 3,000 | Adam optimizer steps |
| Training steps (shared) | 5,000 | Shared SAE training steps |
| Batch size (per-stock) | 256 | Per-stock SAE batch size |
| Batch size (shared) | 512 | Shared SAE batch size |
| Learning rate | 1e-4 | Adam learning rate |
| Train/val/test split | 60/10/30 | Chronological split |
| Tier-1 threshold | \|r\| > 0.15 | Concept discovery |
| Null calibration shuffles | 100 | Permutation null |
| Block size | 10 | Block-permutation calibration |
| Random seed | 42 | Reproducibility |

## Citation

```bibtex
@inproceedings{hou2026interpreting,
  title={Interpreting Financial Time-Series Foundation Models with Sparse Autoencoders: An Empirical Audit of Kronos},
  author={Hou, Wanlong and others},
  booktitle={Proceedings of the 5th ACM International Conference on AI in Finance (ICAIF)},
  year={2026}
}
```

## License

This code is provided for academic research reproducibility. Please cite the paper if you use this code.
