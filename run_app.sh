#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$PROJECT_DIR"

if [ ! -d ".venv" ]; then
  python3 -m venv .venv
fi

source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt

if [ ! -f "data/simulated_shipments.csv" ]; then
  python generate_simulated_data.py
fi

if [ ! -f "models/catboost.joblib" ]; then
  python train_models.py
fi

python -m streamlit run app.py

