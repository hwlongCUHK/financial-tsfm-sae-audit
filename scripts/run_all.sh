#!/bin/bash
# =============================================================================
# SAE Interpretability of Kronos -- Full Reproducibility Pipeline
# =============================================================================
# Run all experiment steps sequentially. Each step depends on the previous.
# Usage: bash scripts/run_all.sh
#
# Prerequisites:
#   - Kronos model weights at the path specified in configs/default.yaml
#   - 120 stock CSVs in the data directory
#   - Python environment with requirements.txt installed
# =============================================================================

set -e  # Exit on first error

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
cd "$PROJECT_DIR"

echo "========================================"
echo "SAE Interpretability Pipeline"
echo "Project root: $PROJECT_DIR"
echo "========================================"
echo ""

echo "Step 1/7: Extract activations..."
echo "  Extracts Kronos layer-6 residual stream activations for all 120 stocks."
python scripts/01_extract_activations.py
echo "  Done."
echo ""

echo "Step 2/7: Train shared SAE..."
echo "  Trains a single SAE on pooled activations from all 120 stocks."
python scripts/02_train_shared_sae.py
echo "  Done."
echo ""

echo "Step 3/7: Concept labeling..."
echo "  Labels SAE features by correlation with 30+ financial statistics."
python scripts/03_concept_labeling.py
echo "  Done."
echo ""

echo "Step 4/7: Ablation and intervention tests..."
echo "  Dose-response ablation, concept-level steering, permutation test."
python scripts/04_ablation_intervention.py
echo "  Done."
echo ""

echo "Step 5/7: Financial validation..."
echo "  Directional accuracy, volatility ratio, RankIC against realized returns."
python scripts/05_financial_validation.py
echo "  Done."
echo ""

echo "Step 6/7: Sensitivity analysis..."
echo "  k and expansion factor sensitivity across 30 stocks."
python scripts/06_sensitivity_analysis.py
echo "  Done."
echo ""

echo "Step 7/7: Generate figures..."
echo "  Produces all publication figures from experiment results."
python scripts/07_visualize.py
echo "  Done."
echo ""

echo "========================================"
echo "Pipeline complete."
echo "Results:  outputs/results/"
echo "Figures:  figures/"
echo "SAE:      outputs/sae/"
echo "========================================"
