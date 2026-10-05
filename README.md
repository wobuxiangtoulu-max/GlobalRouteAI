# GlobalRoute AI ML prototype

This project turns the original rule-based Streamlit dashboard into a trainable delay-prediction prototype. It includes 12,000 reproducible simulated shipments, three model pipelines, chronological evaluation, carrier scenario comparison, model artifacts, and a Streamlit interface.

## What is included

- `data/simulated_shipments.csv`: labelled simulated shipment history.
- `data/simulated_shipments.xlsx`: formatted data workbook with summary and data dictionary.
- `generate_simulated_data.py`: reproducible dataset generator.
- `train_models.py`: Logistic Regression, Random Forest, and CatBoost training.
- `app.py`: dashboard, prediction form, carrier comparison, and model lab.
- `models/`: trained artifacts, holdout predictions, metrics, and feature importance.
- `tests/smoke_test.py`: end-to-end checks for data, training, and recommendation logic.
- `MODEL_CARD.md`: intended use, leakage controls, and limitations.

The bundled records are marked `SIMULATED`. They are suitable for product development and demonstrations, not claims about real logistics performance.

## Fastest way to run on macOS

Open Terminal, enter the extracted project folder, and run:

```bash
bash run_app.sh
```

The script creates a local virtual environment, installs the requirements, trains CatBoost if it is not already available, and opens Streamlit. Keep that Terminal window open while using the app.

## Manual setup

```bash
cd globalroute_ai_ml
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python generate_simulated_data.py
python train_models.py
python -m streamlit run app.py
```

Streamlit normally opens `http://localhost:8501` automatically.

## Training commands

Train all three models:

```bash
python train_models.py
```

Train only selected models:

```bash
python train_models.py --models logistic_regression random_forest
```

Regenerate a different dataset:

```bash
python generate_simulated_data.py --rows 20000 --seed 123
python train_models.py
```

## App workflow

1. Open **Predict and compare**.
2. Enter a planned shipment.
3. Set the financial consequence of a late delivery.
4. Compare the predicted risk, quote, promise, and expected decision cost for every carrier.
5. Open **Model lab** to compare holdout performance and feature importance.
6. Use **Train or refresh all models** after replacing the sample data with a compatible historical CSV.

## Data requirements for real training

Required prediction fields are listed in `data/data_dictionary.csv`. The target is `is_delayed`, where `1` means actual delivery occurred after the promised delivery date.

Before using real data:

- build every feature as it was known at dispatch time;
- exclude actual delivery and post-delivery fields from model inputs;
- preserve chronological splits;
- keep a later untouched validation period;
- replace the simulated cost, promise, and recent-performance functions with actual operational sources.

## Evaluation

The pipeline reports:

- ROC-AUC and PR-AUC;
- Brier score for probability quality;
- precision, recall, and F1 at a 0.50 threshold;
- confusion-matrix counts;
- test start date and test-set delay rate.

PR-AUC is used to select the best available model because delayed shipments are the minority class. In production, choose a probability threshold from the cost of false negatives and false positives rather than accepting 0.50 automatically.

## Smoke test

```bash
python tests/smoke_test.py
```

