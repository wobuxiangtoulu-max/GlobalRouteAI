"""Small end-to-end test that does not require Streamlit or CatBoost."""

from __future__ import annotations

import sys
import tempfile
from datetime import date
from pathlib import Path


PROJECT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_DIR))

from src.modeling import load_artifact, train_all_models  # noqa: E402
from src.recommendation import build_carrier_scenarios  # noqa: E402
from src.simulation import SimulationSettings, generate_shipments  # noqa: E402


def main() -> None:
    df = generate_shipments(
        SimulationSettings(rows=1_200, start_date="2025-01-01", end_date="2026-06-30", seed=7)
    )
    assert len(df) == 1_200
    assert df["shipment_id"].is_unique
    assert df["data_source"].eq("SIMULATED").all()
    assert df["is_delayed"].between(0, 1).all()
    assert 0.10 < df["is_delayed"].mean() < 0.50
    assert (df["delay_days"] == (df["transit_days"] - df["promised_days"]).clip(lower=0)).all()

    with tempfile.TemporaryDirectory() as directory:
        metrics, manifest = train_all_models(
            df,
            output_dir=directory,
            model_names=("logistic_regression", "random_forest"),
            random_state=7,
        )
        assert set(metrics["model_name"]) == {"logistic_regression", "random_forest"}
        assert manifest["test_rows"] > 0
        artifact = load_artifact(Path(directory) / "logistic_regression.joblib")
        comparison = build_carrier_scenarios(
            artifact,
            df,
            ship_date=date(2026, 7, 1),
            origin="Milan",
            destination="Rome",
            service_level="Standard",
            package_weight_kg=4.0,
            package_volume_cm3=5_000.0,
            order_value_eur=120.0,
            fragile=False,
            pickup_hour=14,
            warehouse_load_pct=70.0,
            weather_severity=0,
            holiday_period=False,
            delay_penalty_eur=25.0,
        )
        assert len(comparison) == 5
        assert comparison["recommended"].sum() == 1
        assert comparison["delay_probability"].between(0, 1).all()

    print("Smoke test passed")


if __name__ == "__main__":
    main()

