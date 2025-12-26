#!/bin/bash
# run_dornheim2023.sh
# Run Dornheim 2023 reproduction script for multiple lambda values

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

echo "========================================"
echo "Dornheim 2023 Fictitious Sign Reproduction"
echo "========================================"

# Lambda values from Figure 1
for lambda in 0.0 0.2 0.5 1.0; do
    echo ""
    echo "Running λ_coulomb = $lambda..."
    julia --project="$SCRIPT_DIR/.." "$SCRIPT_DIR/Dornheim2023-Fictitious.jl" $lambda
done

echo ""
echo "========================================"
echo "All runs complete. Generating plots..."
echo "========================================"

gnuplot "$SCRIPT_DIR/Dornheim2023-Fig1.gpt"
gnuplot "$SCRIPT_DIR/Dornheim2023-Fig2.gpt"
gnuplot "$SCRIPT_DIR/Dornheim2023-Fig3.gpt"

echo "Done. Check Dornheim2023-Fig1.pdf, Dornheim2023-Fig2.pdf, Dornheim2023-Fig3.pdf"
