# Auditing Financial Time-Series Foundation Models with Statistics-Grounded Sparse Features

Code repository for the ICAIF 2026 paper: *Auditing Financial Time-Series Foundation Models with Statistics-Grounded Sparse Features*.

> **Status**: Under review at the 5th ACM International Conference on AI in Finance (ICAIF 2026).

## Abstract

Financial time-series foundation models are increasingly deployed in high-stakes applications, yet their internal representations remain opaque. We introduce a reproducible audit protocol that decomposes a financial TSFM's activations using sparse autoencoders (SAEs) and labels the resulting features by correlation with precisely defined financial statistics (momentum, volatility, autocorrelation, tail risk, and volume), using a curated dictionary of 30+ measures to which features are assigned by maximum correlation. Applying this protocol to Kronos (12B K-line records, 120 Chinese A-share stocks), we find that the model's internal concept distribution is remarkably flat: under Tier-1 exploratory labeling ($|r| > 0.15$), 16 concept families are identified with no single family dominating. Ablation experiments confirm that these features measurably affect predictions, primarily by stabilizing output volatility ($+23\%$), with no single concept family dominating prediction responsibility.

## Project Structure

```
.
├── README.md
├── requirements.txt
├── configs/
│   └── default.yaml                    # Experiment hyperparameters
├── src/
│   ├── __init__.py
│   ├── sae.py                          # TopK SAE implementation
│   ├── statistics.py                   # 30+ financial statistics
│   └── utils.py                        # Data loading, normalization
├── scripts/
│   ├── core/                           # Core SAE pipeline (Section 3)
│   │   ├── shared_sae.py               # Shared SAE training (120 stocks pooled)
│   │   ├── shared_sae_120.py           # Shared SAE on full 120 stocks
│   │   ├── train_sae.py                # Per-stock SAE training
│   │   ├── l1_sae.py                   # L1-regularized SAE baseline
│   │   ├── multi_sae.py                # Multi-layer SAE training
│   │   ├── sector_sae.py               # Sector-stratified SAE
│   │   ├── scale120_sae.py             # Scale to 120 stocks
│   │   ├── scale80_sae.py              # Scale to 80 stocks
│   │   ├── scale_to_120.py             # Progressive scaling
│   │   ├── scale_all_to_120.py         # Full-scale training
│   │   ├── scale_eval.py               # Scale evaluation
│   │   ├── multi_stock_eval.py         # Cross-stock evaluation
│   │   ├── fast_retrain.py             # Fast SAE retraining
│   │   ├── run_fast.py                 # Pipeline runner
│   │   └── run_fast2.py                # Pipeline runner v2
│   ├── experiments/                    # exp1--exp9 (Sections 4--6)
│   │   ├── shared_exp_utils.py         # Shared experiment utilities
│   │   ├── exp1_all_token_temporal.py  # All-token temporal analysis
│   │   ├── exp2_multilayer_localization.py  # Multi-layer concept localization
│   │   ├── exp3_pooled_ordinal.py      # Pooled ordinal analysis
│   │   ├── exp4_sequence_aggregation.py    # Sequence aggregation
│   │   ├── exp5_cross_validated_tcav.py    # Cross-validated TCAV
│   │   ├── exp6_nonlinear_dependence.py    # Nonlinear dependence
│   │   ├── exp7_token_counterfactual.py    # Token counterfactuals
│   │   ├── exp8_heldout_label_recovery.py  # Held-out label recovery
│   │   └── exp9_correlation_profile.py     # Correlation profile stability
│   ├── analysis/                       # Statistical analysis
│   │   ├── block_bootstrap.py          # Block bootstrap (10-window blocks)
│   │   ├── block_permutation.py        # Block-permutation calibration
│   │   ├── clustered_bootstrap.py      # Sector-clustered bootstrap
│   │   ├── permutation_test.py         # Permutation intervention test
│   │   ├── perm_clean.py               # Clean permutation analysis
│   │   ├── stability_check.py          # SAE stability across seeds/layers
│   │   ├── pca_baseline.py             # PCA baseline comparison
│   │   ├── pooled_calibration.py       # Pooled null calibration
│   │   ├── recalibrate_null.py         # Null recalibration
│   │   ├── fast_null.py                # Efficient null computation
│   │   ├── counterfactual_selectivity.py    # Counterfactual window pairs
│   │   ├── cross_stock_confirmation.py      # Cross-stock consistency
│   │   ├── event_localization.py            # Event-driven analysis
│   │   ├── family_level_decoding.py         # Concept family decoding
│   │   └── sector_consist.py                # Sector consistency
│   ├── financial/                      # Financial metrics (Section 5)
│   │   ├── financial_impact.py         # Financial impact assessment
│   │   ├── financial_metrics.py        # Core financial metrics
│   │   ├── fin_metrics_v2.py           # Extended financial metrics v2
│   │   ├── fin_metrics_expanded.py     # Expanded financial validation
│   │   ├── ground_truth_120.py         # Ground-truth on 120 stocks
│   │   └── ground_truth_metrics.py     # Ground-truth metric computation
│   ├── ablation/                       # Intervention & ablation (Section 5)
│   │   ├── factor_intervention.py      # Factor-level interventions
│   │   ├── importance_matched_control.py    # Importance-matched baselines
│   │   ├── rich_features_steer.py      # Feature steering analysis
│   │   ├── steering_gen.py             # Steering generalization
│   │   ├── direct_cvar.py              # Direct CVaR intervention
│   │   └── concept_vs_random.py        # Concept vs random comparison
│   ├── sensitivity/                    # Sensitivity analysis (Section 5.5)
│   │   └── sensitivity_analysis.py     # k/expansion sensitivity
│   ├── visualization/                  # Figure generation
│   │   ├── visualize_activations.py    # Activation heatmaps
│   │   └── visualize_more_stocks.py    # Extended stock visualizations
│   └── utils/                          # Utilities
│       ├── analyze_attn.py             # Attention analysis
│       ├── apply_fixes.py              # Reviewer fix application
│       ├── save_results.py             # Result serialization
│       ├── show_results.py             # Result display
│       └── check_experiments.sh        # Experiment status check
├── outputs/                            # Experiment results (JSON)
│   └── sae/                            # 68 result files
└── data/                               # Data directory (not tracked)
    └── scale120/                        # 120 A-share stock CSVs
```

## Data Requirements

1. **Kronos model**: Kronos-base weights and tokenizer.
2. **Stock data**: 120 Chinese A-share stock CSVs from Shanghai/Shenzhen exchanges (2005--2022). Four sectors: Bank, Energy, Technology, Consumer (30 each).

## Quickstart

```bash
pip install -r requirements.txt

# 1. Extract Kronos activations
python scripts/extraction/run_extract.sh

# 2. Train shared SAE
python scripts/core/shared_sae_120.py

# 3. Run concept labeling
python scripts/experiments/exp8_heldout_label_recovery.py

# 4. Run ablation experiments
python scripts/ablation/factor_intervention.py

# 5. Run financial metric validation
python scripts/financial/fin_metrics_expanded.py

# 6. Run all experiments (exp1--exp9)
for i in {1..9}; do python scripts/experiments/exp${i}_*.py; done
```

## Key Hyperparameters

| Parameter | Value | Description |
|---|---|---|
| Model | Kronos-base | 12 layers, 832-dim, 102M params |
| Layer | 6 (primary), {0,3,6,9,11} (multilayer) | Residual stream extraction |
| Window length | 64 | K-line periods per window |
| Stride | 32 | Sliding window stride |
| SAE k | 64 | Active features per sample |
| SAE expansion | 4x | d_hidden = 832 * 4 = 3328 |
| Training steps (shared) | 5,000 | Adam, lr=1e-4 |
| Training steps (per-stock) | 3,000 | Adam, lr=1e-4 |
| Batch size (shared) | 512 | Per-stock: 256 |
| Train/purge/test split | 60/10/30 | Chronological |
| Tier-1 threshold | $|r| > 0.15$ | Concept discovery |
| Null shuffles | 100 | Permutation calibration |
| Block size | 10 | Block-permutation |
| Random seed | 42 | Reproducibility |
| Precision | FP16 | Single NVIDIA GPU |

## Citation

```bibtex
@inproceedings{hou2026auditing,
  title={Auditing Financial Time-Series Foundation Models with Statistics-Grounded Sparse Features},
  author={Hou, Wanlong and others},
  booktitle={Proceedings of the 5th ACM International Conference on AI in Finance (ICAIF)},
  year={2026}
}
```

## License

This code is provided for academic research reproducibility. Please cite the paper if you use this code.
