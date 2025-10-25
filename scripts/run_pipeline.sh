#!/bin/bash
# scripts/run_pipeline.sh

set -e

echo "========================================" echo "Running Colorectal Cancer Classification Pipeline"
echo "========================================"

# Activate environment
if [ -f "environment.yml" ]; then
    echo "Activating conda environment..."
    conda env create -f environment.yml || true
    conda activate colorectal_cancer
else
    echo "Installing requirements..."
    pip install -r requirements.txt
fi

# Run pipeline
echo "Starting main pipeline..."
python main.py

echo "========================================echo "Pipeline completed successfully!"
echo "========================================"
