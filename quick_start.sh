#!/bin/bash
# Quick Start Script for RTAI Lab 5 - Air Quality Monitoring System

echo "=========================================="
echo "RTAI Lab 5: Air Quality Monitoring System"
echo "=========================================="
echo ""

# Check Python
echo "✓ Checking Python installation..."
python3 --version

# Install dependencies
echo ""
echo "📦 Installing dependencies..."
python3 -m pip install --quiet -r requirements.txt
echo "✓ Dependencies installed"

# Fetch data
echo ""
echo "🌍 Fetching real air quality data..."
python3 data_acquisition.py
echo "✓ Data downloaded"

# Train model
echo ""
echo "🤖 Training ML model and generating visualizations..."
python3 train_model.py
echo "✓ Model trained"

# Final summary
echo ""
echo "=========================================="
echo "✅ SETUP COMPLETE!"
echo "=========================================="
echo ""
echo "📊 Files created:"
echo "  • data/ - 3 CSV files with real air quality data"
echo "  • models/ - Trained Random Forest model"
echo "  • visualizations/ - 8 PNG visualizations"
echo ""
echo "🚀 To launch the dashboard, run:"
echo "   streamlit run dashboard.py"
echo ""
echo "📖 For detailed documentation, see:"
echo "   • README.md - User guide"
echo "   • REPORT.md - Technical report"
echo "   • SUBMISSION_SUMMARY.md - Deliverables checklist"
echo ""
echo "=========================================="
