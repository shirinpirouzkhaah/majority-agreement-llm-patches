#!/usr/bin/env bash

set -euo pipefail

# Run all 3 models across all 5 similarity metrics.
#
# Environment mapping:
#   bleu       -> no conda environment activation
#   codebleu   -> codebleu-clustering
#   codebleu2  -> codebleu-clustering
#   codebert   -> codebert-clustering
#   voyage     -> voyage-clustering

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON_SCRIPT="$SCRIPT_DIR/clustering_analysis_revised.py"

MODELS=(
    qwen
    gpt
    claude
)

run_metric_for_all_models() {
    local metric="$1"

    for model in "${MODELS[@]}"; do
        echo
        echo "============================================================"
        echo "Running model=${model}, metric=${metric}"
        echo "============================================================"

        python "$PYTHON_SCRIPT" \
            --model "$model" \
            --metric "$metric"
    done
}

# ---------------------------------------------------------------------------
# BLEU
# Run before activating any Conda environment.
# ---------------------------------------------------------------------------
echo
echo "################################################################"
echo "BLEU: running without activating a Conda environment"
echo "################################################################"

run_metric_for_all_models "bleu"


# ---------------------------------------------------------------------------
# Initialize Conda shell support for the remaining metrics.
# This does not activate an environment by itself.
# ---------------------------------------------------------------------------
if ! command -v conda >/dev/null 2>&1; then
    echo "ERROR: conda was not found in PATH." >&2
    exit 1
fi

eval "$(conda shell.bash hook)"


# ---------------------------------------------------------------------------
# CodeBLEU and CodeBLEU2
# ---------------------------------------------------------------------------
echo
echo "################################################################"
echo "Activating codebleu-clustering for CodeBLEU and CodeBLEU2"
echo "################################################################"

conda activate codebleu-clustering

run_metric_for_all_models "codebleu"
run_metric_for_all_models "codebleu2"


# ---------------------------------------------------------------------------
# CodeBERT
# ---------------------------------------------------------------------------
echo
echo "################################################################"
echo "Activating codebert-clustering for CodeBERT"
echo "################################################################"

conda activate codebert-clustering

run_metric_for_all_models "codebert"


# ---------------------------------------------------------------------------
# Voyage
# ---------------------------------------------------------------------------
echo
echo "################################################################"
echo "Activating voyage-clustering for Voyage"
echo "################################################################"

conda activate voyage-clustering

run_metric_for_all_models "voyage"


echo
echo "============================================================"
echo "All 15 model/metric runs completed successfully."
echo "============================================================"
