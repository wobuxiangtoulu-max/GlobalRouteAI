"""Streamlit interface for GlobalRoute AI delay prediction and carrier comparison."""

from __future__ import annotations

import json
from datetime import timedelta
from pathlib import Path

import pandas as pd
import streamlit as st

from src.config import CARRIERS, DESTINATIONS, ORIGINS, SERVICE_LEVELS
from src.features import validate_source_data
from src.modeling import MODEL_DISPLAY_NAMES, load_artifact, train_all_models
from src.recommendation import build_carrier_scenarios
from src.simulation import data_dictionary


BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "data" / "simulated_shipments.csv"
MODEL_DIR = BASE_DIR / "models"

st.set_page_config(page_title="GlobalRoute AI", page_icon="🚚", layout="wide")


@st.cache_data
def load_default_data() -> pd.DataFrame:
    return pd.read_csv(DATA_PATH)


@st.cache_resource
def load_model(model_name: str, modified_time: float):
    del modified_time
    return load_artifact(MODEL_DIR / f"{model_name}.joblib")


def available_models() -> list[str]:
    return [name for name in MODEL_DISPLAY_NAMES if (MODEL_DIR / f"{name}.joblib").exists()]


def read_manifest() -> dict:
    path = MODEL_DIR / "model_manifest.json"
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}


def load_history() -> tuple[pd.DataFrame, str]:
    uploaded = st.sidebar.file_uploader("Upload enhanced shipment CSV", type=["csv"])
    if uploaded is None:
        return load_default_data(), "Bundled simulated dataset"
    return pd.read_csv(uploaded), uploaded.name


def percentage(value: float) -> str:
    return f"{100 * value:.1f}%"


history, data_label = load_history()
try:
    validate_source_data(history, require_target=True)
except ValueError as exc:
    st.error(str(exc))
    st.stop()

history["ship_date"] = pd.to_datetime(history["ship_date"], errors="coerce")
if history["ship_date"].isna().any():
    st.error("ship_date contains invalid values.")
    st.stop()

st.sidebar.header("Model controls")
models = available_models()
if st.sidebar.button("Train or refresh all models", use_container_width=True):
    with st.spinner("Training models with a chronological holdout..."):
        try:
            train_all_models(history, MODEL_DIR)
            st.cache_resource.clear()
            st.success("Training complete.")
            st.rerun()
        except Exception as exc:
            st.sidebar.error(f"Training failed: {exc}")

models = available_models()
if not models:
    st.error("No trained model was found. Use the sidebar training button or run `python train_models.py`.")
    st.stop()

manifest = read_manifest()
default_model = manifest.get("best_available_model", models[0])
if default_model not in models:
    default_model = models[0]
selected_model = st.sidebar.selectbox(
    "Prediction model",
    models,
    index=models.index(default_model),
    format_func=lambda name: MODEL_DISPLAY_NAMES[name],
)
model_path = MODEL_DIR / f"{selected_model}.joblib"
artifact = load_model(selected_model, model_path.stat().st_mtime)

st.title("GlobalRoute AI")
st.caption("Shipment delay prediction and carrier decision prototype")
if "data_source" in history.columns and (history["data_source"] == "SIMULATED").all():
    st.warning(
        "This build uses simulated data. Model scores demonstrate the pipeline, not validated commercial performance."
    )

tab_predict, tab_dashboard, tab_models, tab_data = st.tabs(
    ["Predict and compare", "Operations dashboard", "Model lab", "Data"]
)

with tab_predict:
    st.subheader("Compare carriers before dispatch")
    st.write(
        "The selected model scores the same planned shipment once for each carrier. "
        "The recommendation minimises quote plus expected delay penalty."
    )
    max_history_date = history["ship_date"].max().date()
    with st.form("shipment_scenario"):
        row1 = st.columns(4)
        ship_date = row1[0].date_input("Ship date", value=max_history_date + timedelta(days=1))
        origin = row1[1].selectbox("Origin", ORIGINS)
        destination_options = [city for city in DESTINATIONS if city != origin]
        destination = row1[2].selectbox("Destination", destination_options, index=min(1, len(destination_options) - 1))
        service_level = row1[3].selectbox("Service level", SERVICE_LEVELS, index=1)

        row2 = st.columns(4)
        weight = row2[0].number_input("Weight (kg)", min_value=0.1, max_value=100.0, value=4.0, step=0.5)
        volume = row2[1].number_input(
            "Volume (cm³)", min_value=100.0, max_value=200_000.0, value=5_000.0, step=500.0
        )
        order_value = row2[2].number_input(
            "Order value (€)", min_value=1.0, max_value=50_000.0, value=120.0, step=10.0
        )
        fragile = row2[3].checkbox("Fragile shipment", value=False)

        row3 = st.columns(5)
        pickup_hour = row3[0].slider("Pickup hour", 8, 19, 14)
        warehouse_load = row3[1].slider("Warehouse load (%)", 35, 100, 70)
        weather = row3[2].select_slider("Weather severity", options=[0, 1, 2, 3], value=0)
        holiday = row3[3].checkbox("Holiday period", value=False)
        delay_penalty = row3[4].number_input(
            "Delay penalty (€)", min_value=0.0, max_value=500.0, value=25.0, step=5.0
        )
        submitted = st.form_submit_button("Compare all carriers", use_container_width=True)

    if submitted:
        comparison = build_carrier_scenarios(
            artifact,
            history,
            ship_date=ship_date,
            origin=origin,
            destination=destination,
            service_level=service_level,
            package_weight_kg=float(weight),
            package_volume_cm3=float(volume),
            order_value_eur=float(order_value),
            fragile=fragile,
            pickup_hour=pickup_hour,
            warehouse_load_pct=float(warehouse_load),
            weather_severity=weather,
            holiday_period=holiday,
            delay_penalty_eur=float(delay_penalty),
        )
        st.session_state["carrier_comparison"] = comparison
        st.session_state["carrier_comparison_model"] = selected_model

    comparison = st.session_state.get("carrier_comparison")
    comparison_model = st.session_state.get("carrier_comparison_model")
    if comparison is not None and comparison_model != selected_model:
        st.info("The selected model changed. Submit the shipment again to refresh the comparison.")
    elif comparison is not None:
        recommendation = comparison.loc[comparison["recommended"]].iloc[0]
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Recommended carrier", recommendation["carrier"])
        m2.metric("Predicted delay risk", percentage(float(recommendation["delay_probability"])))
        m3.metric("Quoted shipping cost", f"€{recommendation['shipping_cost']:.2f}")
        m4.metric("Promised transit", f"{int(recommendation['promised_days'])} days")

        display = comparison[
            [
                "carrier",
                "delay_probability",
                "risk_level",
                "shipping_cost",
                "promised_days",
                "carrier_recent_on_time_rate",
                "expected_decision_cost",
                "recommended",
            ]
        ].copy()
        display["delay_probability"] = display["delay_probability"].map(percentage)
        display["carrier_recent_on_time_rate"] = display["carrier_recent_on_time_rate"].map(percentage)
        display["shipping_cost"] = display["shipping_cost"].map(lambda value: f"€{value:.2f}")
        display["expected_decision_cost"] = display["expected_decision_cost"].map(lambda value: f"€{value:.2f}")
        display = display.rename(
            columns={
                "carrier": "Carrier",
                "delay_probability": "Delay risk",
                "risk_level": "Risk level",
                "shipping_cost": "Quote",
                "promised_days": "Promised days",
                "carrier_recent_on_time_rate": "Recent on-time rate",
                "expected_decision_cost": "Expected decision cost",
                "recommended": "Recommended",
            }
        )
        st.dataframe(display, use_container_width=True, hide_index=True)
        chart_data = comparison.set_index("carrier")[["delay_probability"]]
        st.bar_chart(chart_data, y_label="Predicted delay probability")
        st.caption(
            "Expected decision cost = quoted cost + predicted delay probability × delay penalty. "
            "Change the penalty to reflect the value of a late delivery."
        )

with tab_dashboard:
    st.subheader("Historical operations")
    total = len(history)
    delay_rate = float(history["is_delayed"].mean())
    average_cost = float(pd.to_numeric(history["shipping_cost"]).mean())
    average_transit = (
        float(pd.to_numeric(history["transit_days"]).mean()) if "transit_days" in history.columns else float("nan")
    )
    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Shipments", f"{total:,}")
    k2.metric("Delay rate", percentage(delay_rate))
    k3.metric("Average shipping cost", f"€{average_cost:.2f}")
    k4.metric("Average transit", f"{average_transit:.2f} days" if pd.notna(average_transit) else "n.a.")

    carrier_aggregations = {
        "shipments": ("carrier", "size"),
        "delay_rate": ("is_delayed", "mean"),
        "average_cost": ("shipping_cost", "mean"),
    }
    if "transit_days" in history.columns:
        carrier_aggregations["average_transit"] = ("transit_days", "mean")
    carrier_summary = history.groupby("carrier", as_index=False).agg(**carrier_aggregations).sort_values("delay_rate")
    carrier_display = carrier_summary.copy()
    carrier_display["delay_rate"] = carrier_display["delay_rate"].map(percentage)
    carrier_display["average_cost"] = carrier_display["average_cost"].map(lambda value: f"€{value:.2f}")
    if "average_transit" in carrier_display.columns:
        carrier_display["average_transit"] = carrier_display["average_transit"].round(2)
    st.markdown("#### Carrier performance")
    st.dataframe(carrier_display, use_container_width=True, hide_index=True)

    monthly = history.assign(month=history["ship_date"].dt.to_period("M").astype(str)).groupby("month").agg(
        delay_rate=("is_delayed", "mean"), shipments=("carrier", "size")
    )
    st.markdown("#### Monthly delay rate")
    st.line_chart(monthly[["delay_rate"]], y_label="Delay rate")

with tab_models:
    st.subheader("Model comparison")
    metrics_path = MODEL_DIR / "model_metrics.csv"
    if metrics_path.exists():
        metrics = pd.read_csv(metrics_path)
        metric_columns = [
            "display_name",
            "roc_auc",
            "pr_auc",
            "brier_score",
            "precision",
            "recall",
            "f1",
            "train_rows",
            "test_rows",
            "test_start_date",
        ]
        st.dataframe(metrics[metric_columns], use_container_width=True, hide_index=True)
        st.caption(
            "All scores use the newest 20% of shipments as a chronological test set. "
            "PR-AUC is the selection metric because delay events are the minority class."
        )
    skipped = manifest.get("skipped_models", {})
    if skipped:
        for name, reason in skipped.items():
            st.info(f"{MODEL_DISPLAY_NAMES.get(name, name)} is not yet available: {reason}")

    importance_path = MODEL_DIR / f"feature_importance_{selected_model}.csv"
    if importance_path.exists():
        importance = pd.read_csv(importance_path).head(15)
        st.markdown(f"#### Main features: {MODEL_DISPLAY_NAMES[selected_model]}")
        st.bar_chart(importance.set_index("feature")[["importance"]], horizontal=True)
        st.dataframe(importance, use_container_width=True, hide_index=True)

with tab_data:
    st.subheader("Dataset")
    st.write(f"Active source: **{data_label}**")
    st.caption(
        "Outcome fields such as delivery_date, transit_days, delay_days, and delay_reason are excluded from prediction."
    )
    st.dataframe(history.head(500), use_container_width=True, hide_index=True)
    st.markdown("#### Data dictionary")
    st.dataframe(data_dictionary(), use_container_width=True, hide_index=True)
