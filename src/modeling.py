"""Train, evaluate, persist, and score delay-classification models."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from .config import RANDOM_SEED
from .features import CATEGORICAL_FEATURES, MODEL_FEATURES, NUMERIC_FEATURES, prepare_features


MODEL_DISPLAY_NAMES = {
    "logistic_regression": "Logistic Regression",
    "random_forest": "Random Forest",
    "catboost": "CatBoost",
}


def chronological_split(
    df: pd.DataFrame,
    test_fraction: float = 0.20,
) -> tuple[pd.DataFrame, pd.DataFrame, str]:
    if not 0.10 <= test_fraction <= 0.40:
        raise ValueError("test_fraction must be between 0.10 and 0.40")
    dated = df.copy()
    dated["ship_date"] = pd.to_datetime(dated["ship_date"], errors="raise")
    dated = dated.sort_values("ship_date", kind="stable").reset_index(drop=True)
    cutoff_position = min(max(int(len(dated) * (1.0 - test_fraction)), 1), len(dated) - 1)
    cutoff_date = dated.loc[cutoff_position, "ship_date"]
    train = dated[dated["ship_date"] < cutoff_date].copy()
    test = dated[dated["ship_date"] >= cutoff_date].copy()
    if train.empty or test.empty:
        raise ValueError("Chronological split produced an empty train or test set")
    return train, test, cutoff_date.strftime("%Y-%m-%d")


def _preprocessor(scale_numeric: bool) -> ColumnTransformer:
    numeric_steps: list[tuple[str, Any]] = [("imputer", SimpleImputer(strategy="median"))]
    if scale_numeric:
        numeric_steps.append(("scaler", StandardScaler()))
    numeric_pipeline = Pipeline(numeric_steps)
    categorical_pipeline = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore", min_frequency=5)),
        ]
    )
    return ColumnTransformer(
        [
            ("numeric", numeric_pipeline, NUMERIC_FEATURES),
            ("categorical", categorical_pipeline, CATEGORICAL_FEATURES),
        ],
        remainder="drop",
    )


def build_sklearn_model(model_name: str, random_state: int = RANDOM_SEED) -> Pipeline:
    if model_name == "logistic_regression":
        estimator = LogisticRegression(
            C=0.85,
            max_iter=2_000,
            solver="lbfgs",
            random_state=random_state,
        )
        return Pipeline([("preprocess", _preprocessor(scale_numeric=True)), ("model", estimator)])
    if model_name == "random_forest":
        estimator = RandomForestClassifier(
            n_estimators=240,
            max_depth=14,
            min_samples_leaf=6,
            max_features="sqrt",
            n_jobs=-1,
            random_state=random_state,
        )
        return Pipeline([("preprocess", _preprocessor(scale_numeric=False)), ("model", estimator)])
    raise ValueError(f"Unsupported scikit-learn model: {model_name}")


def evaluate_binary_classifier(y_true: pd.Series, probability: np.ndarray, threshold: float = 0.50) -> dict[str, float | int]:
    prediction = (probability >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, prediction, labels=[0, 1]).ravel()
    return {
        "roc_auc": round(float(roc_auc_score(y_true, probability)), 4),
        "pr_auc": round(float(average_precision_score(y_true, probability)), 4),
        "brier_score": round(float(brier_score_loss(y_true, probability)), 4),
        "accuracy": round(float(accuracy_score(y_true, prediction)), 4),
        "precision": round(float(precision_score(y_true, prediction, zero_division=0)), 4),
        "recall": round(float(recall_score(y_true, prediction, zero_division=0)), 4),
        "f1": round(float(f1_score(y_true, prediction, zero_division=0)), 4),
        "true_negative": int(tn),
        "false_positive": int(fp),
        "false_negative": int(fn),
        "true_positive": int(tp),
        "threshold": threshold,
    }


def _sklearn_feature_importance(model_name: str, pipeline: Pipeline) -> pd.DataFrame:
    feature_names = pipeline.named_steps["preprocess"].get_feature_names_out()
    estimator = pipeline.named_steps["model"]
    if model_name == "logistic_regression":
        signed = estimator.coef_[0]
        frame = pd.DataFrame(
            {
                "feature": feature_names,
                "importance": np.abs(signed),
                "direction": np.where(signed >= 0, "higher delay risk", "lower delay risk"),
                "signed_effect": signed,
            }
        )
    else:
        frame = pd.DataFrame(
            {
                "feature": feature_names,
                "importance": estimator.feature_importances_,
                "direction": "non-directional",
                "signed_effect": np.nan,
            }
        )
    frame["feature"] = frame["feature"].str.replace("numeric__", "", regex=False).str.replace(
        "categorical__", "", regex=False
    )
    return frame.sort_values("importance", ascending=False).reset_index(drop=True)


def _prepare_catboost_frame(
    X: pd.DataFrame,
    numeric_fill_values: dict[str, float] | None = None,
) -> tuple[pd.DataFrame, dict[str, float]]:
    prepared = X[MODEL_FEATURES].copy()
    for column in CATEGORICAL_FEATURES:
        prepared[column] = prepared[column].fillna("__MISSING__").astype(str)
    if numeric_fill_values is None:
        numeric_fill_values = {
            column: float(pd.to_numeric(prepared[column], errors="coerce").median())
            for column in NUMERIC_FEATURES
        }
    for column in NUMERIC_FEATURES:
        prepared[column] = pd.to_numeric(prepared[column], errors="coerce").fillna(numeric_fill_values[column])
    return prepared, numeric_fill_values


def _train_catboost(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    random_state: int,
) -> tuple[Any, dict[str, float], pd.DataFrame]:
    try:
        from catboost import CatBoostClassifier
    except ImportError as exc:
        raise RuntimeError("CatBoost is not installed. Run: pip install -r requirements.txt") from exc

    prepared, fill_values = _prepare_catboost_frame(X_train)
    categorical_indexes = [prepared.columns.get_loc(column) for column in CATEGORICAL_FEATURES]
    model = CatBoostClassifier(
        iterations=500,
        depth=7,
        learning_rate=0.055,
        loss_function="Logloss",
        eval_metric="AUC",
        random_seed=random_state,
        verbose=False,
        allow_writing_files=False,
        thread_count=-1,
    )
    model.fit(prepared, y_train, cat_features=categorical_indexes)
    importance = pd.DataFrame(
        {
            "feature": MODEL_FEATURES,
            "importance": model.get_feature_importance(),
            "direction": "non-directional",
            "signed_effect": np.nan,
        }
    ).sort_values("importance", ascending=False, ignore_index=True)
    return model, fill_values, importance


def predict_probability(artifact: dict[str, Any], source_rows: pd.DataFrame) -> np.ndarray:
    X, _ = prepare_features(source_rows, require_target=False)
    if artifact["model_name"] == "catboost":
        prepared, _ = _prepare_catboost_frame(X, artifact["numeric_fill_values"])
        return artifact["estimator"].predict_proba(prepared)[:, 1]
    return artifact["estimator"].predict_proba(X)[:, 1]


def load_artifact(path: str | Path) -> dict[str, Any]:
    return joblib.load(Path(path))


def train_all_models(
    df: pd.DataFrame,
    output_dir: str | Path,
    model_names: Iterable[str] = ("logistic_regression", "random_forest", "catboost"),
    test_fraction: float = 0.20,
    random_state: int = RANDOM_SEED,
    require_catboost: bool = False,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Train requested models and persist comparable artifacts and test predictions."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    train_df, test_df, cutoff_date = chronological_split(df, test_fraction=test_fraction)
    X_train, y_train = prepare_features(train_df, require_target=True)
    X_test, y_test = prepare_features(test_df, require_target=True)
    trained_at = datetime.now(timezone.utc).isoformat()

    metrics_rows: list[dict[str, Any]] = []
    skipped: dict[str, str] = {}
    prediction_frame = pd.DataFrame(
        {
            "shipment_id": test_df.get("shipment_id", pd.Series(range(len(test_df)))).astype(str).to_numpy(),
            "ship_date": pd.to_datetime(test_df["ship_date"]).dt.strftime("%Y-%m-%d").to_numpy(),
            "actual_is_delayed": y_test.to_numpy(),
        }
    )

    for model_name in model_names:
        if model_name not in MODEL_DISPLAY_NAMES:
            raise ValueError(f"Unknown model: {model_name}")
        numeric_fill_values: dict[str, float] | None = None
        try:
            if model_name == "catboost":
                estimator, numeric_fill_values, importance = _train_catboost(X_train, y_train, random_state)
                artifact = {
                    "model_name": model_name,
                    "display_name": MODEL_DISPLAY_NAMES[model_name],
                    "estimator": estimator,
                    "numeric_fill_values": numeric_fill_values,
                }
                probability = predict_probability(artifact, test_df)
            else:
                estimator = build_sklearn_model(model_name, random_state=random_state)
                estimator.fit(X_train, y_train)
                probability = estimator.predict_proba(X_test)[:, 1]
                importance = _sklearn_feature_importance(model_name, estimator)
                artifact = {
                    "model_name": model_name,
                    "display_name": MODEL_DISPLAY_NAMES[model_name],
                    "estimator": estimator,
                }
        except RuntimeError as exc:
            if model_name == "catboost" and not require_catboost:
                skipped[model_name] = str(exc)
                continue
            raise

        metrics = evaluate_binary_classifier(y_test, probability)
        metrics.update(
            {
                "model_name": model_name,
                "display_name": MODEL_DISPLAY_NAMES[model_name],
                "train_rows": len(train_df),
                "test_rows": len(test_df),
                "train_delay_rate": round(float(y_train.mean()), 4),
                "test_delay_rate": round(float(y_test.mean()), 4),
                "test_start_date": cutoff_date,
                "trained_at_utc": trained_at,
            }
        )
        artifact.update(
            {
                "model_features": MODEL_FEATURES,
                "categorical_features": CATEGORICAL_FEATURES,
                "numeric_features": NUMERIC_FEATURES,
                "metrics": metrics,
                "test_start_date": cutoff_date,
                "trained_at_utc": trained_at,
                "random_state": random_state,
            }
        )
        joblib.dump(artifact, output_dir / f"{model_name}.joblib")
        importance.head(50).to_csv(output_dir / f"feature_importance_{model_name}.csv", index=False)
        prediction_frame[f"probability_{model_name}"] = probability
        metrics_rows.append(metrics)

    if not metrics_rows:
        raise RuntimeError("No model was trained")

    metrics_df = pd.DataFrame(metrics_rows).sort_values("pr_auc", ascending=False).reset_index(drop=True)
    metrics_df.to_csv(output_dir / "model_metrics.csv", index=False)
    prediction_frame.to_csv(output_dir / "test_predictions.csv", index=False)
    best_model = str(metrics_df.iloc[0]["model_name"])
    manifest = {
        "trained_at_utc": trained_at,
        "test_start_date": cutoff_date,
        "selection_metric": "pr_auc",
        "best_available_model": best_model,
        "available_models": metrics_df["model_name"].tolist(),
        "skipped_models": skipped,
        "dataset_rows": int(len(df)),
        "train_rows": int(len(train_df)),
        "test_rows": int(len(test_df)),
        "simulated_data": bool("data_source" in df.columns and (df["data_source"] == "SIMULATED").all()),
    }
    (output_dir / "model_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return metrics_df, manifest

