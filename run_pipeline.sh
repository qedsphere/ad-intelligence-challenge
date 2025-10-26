#!/bin/bash
# Convenient script to clean output and run the optimized pipeline

set -e  # Exit on error

echo "╔════════════════════════════════════════════════════════════════╗"
echo "║          Audio Feature Extraction Pipeline - Clean Run        ║"
echo "╚════════════════════════════════════════════════════════════════╝"
echo ""

# Navigate to script directory
cd "$(dirname "$0")"

# Clean previous outputs
echo "🧹 Cleaning previous outputs..."
rm -rf output/vocals
rm -f output/features.json
echo "   ✓ Cleaned: output/vocals/"
echo "   ✓ Cleaned: output/features.json"
echo ""

# Activate virtual environment
echo "🔧 Activating virtual environment..."
source .venv/bin/activate
echo "   ✓ Virtual environment active"
echo ""

# Run the pipeline
echo "🚀 Starting pipeline..."
echo ""
time python main.py --clean --progress

echo ""
echo "╔════════════════════════════════════════════════════════════════╗"
echo "║                       Pipeline Complete! ✓                     ║"
echo "╚════════════════════════════════════════════════════════════════╝"
echo ""
echo "📊 Results saved to: output/features.json"
echo ""
echo "To view results:"
echo "  cat output/features.json | jq '.'"
echo ""


