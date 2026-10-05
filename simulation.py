"""Generate reproducible, explicitly simulated Italian parcel shipments."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from math import asin, cos, radians, sin, sqrt
from pathlib import Path

import numpy as np
import pandas as pd

from .config import (
    CARRIERS,
    CARRIER_BASE_COST,
    CARRIER_BASE_ON_TIME_RATE,
    CARRIER_WEIGHTS,
    CITY_COORDINATES,
    DESTINATIONS,
    DESTINATION_WEIGHTS,
    ISLAND_DESTINATIONS,
    ORIGINS,
    ORIGIN_WEIGHTS,
    RANDOM_SEED,
    SERVICE_COST_MULTIPLIER,
    SERVICE_LEVELS,
    SERVICE_WEIGHTS,
    SOUTHERN_DESTINATIONS,
)


@dataclass(frozen=True)
class SimulationSettings:
    rows: int = 12_000
    start_date: str = "2025-01-01"
    end_date: str = "2026-09-30"
    seed: int = RANDOM_SEED


def _sigmoid(value: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-value))


def estimate_distance_km(origin: str, destination: str) -> int:
    """Estimate driving distance from city coordinates, including island handling."""
    lat1, lon1 = CITY_COORDINATES[origin]
    lat2, lon2 = CITY_COORDINATES[destination]
    radius_km = 6371.0
    dlat = radians(lat2 - lat1)
    dlon = radians(lon2 - lon1)
    a = sin(dlat / 2) ** 2 + cos(radians(lat1)) * cos(radians(lat2)) * sin(dlon / 2) ** 2
    air_distance = 2 * radius_km * asin(sqrt(a))
    road_factor = 1.18
    ferry_allowance = 115 if destination in ISLAND_DESTINATIONS else 0
    return max(35, int(round(air_distance * road_factor + ferry_allowance)))


def calculate_promised_days(
    origin: str,
    destination: str,
    service_level: str,
    carrier: str,
    distance_km: int | None = None,
) -> int:
    """Return a deterministic promised transit time known before dispatch."""
    distance = distance_km or estimate_distance_km(origin, destination)
    base_days = int(np.ceil(distance / 430.0))
    if destination in ISLAND_DESTINATIONS:
        base_days += 1
    service_adjustment = {"Express": -1, "Standard": 0, "Economy": 1}[service_level]
    carrier_adjustment = {"DHL": 0, "UPS": 0, "GLS": 0, "BRT": 0, "Poste Italiane": 1}[carrier]
    return int(np.clip(base_days + service_adjustment + carrier_adjustment, 1, 7))


def quote_shipping_cost(
    carrier: str,
    service_level: str,
    weight_kg: float,
    volume_cm3: float,
    distance_km: int,
    fragile: bool,
    noise_eur: float = 0.0,
) -> float:
    """Create a repeatable pre-shipment quote used by the simulator and app."""
    volumetric_weight = volume_cm3 / 5000.0
    billable_weight = max(weight_kg, volumetric_weight)
    subtotal = (
        CARRIER_BASE_COST[carrier]
        + 0.0095 * distance_km
        + 0.31 * max(billable_weight - 1.0, 0.0)
        + (1.20 if fragile else 0.0)
    )
    return round(max(4.5, subtotal * SERVICE_COST_MULTIPLIER[service_level] + noise_eur), 2)


def _holiday_period(ship_dates: pd.DatetimeIndex) -> np.ndarray:
    month = ship_dates.month.to_numpy()
    day = ship_dates.day.to_numpy()
    christmas = ((month == 12) & (day >= 10)) | ((month == 1) & (day <= 7))
    summer = (month == 8) & (day >= 5) & (day <= 25)
    easter_approximation = (month == 4) & (day >= 10) & (day <= 20)
    return christmas | summer | easter_approximation


def _choose_delay_reason(
    rng: np.random.Generator,
    delayed: bool,
    weather_severity: int,
    warehouse_load_pct: float,
    holiday_period: bool,
    destination: str,
) -> str:
    if not delayed:
        return "No delay"
    labels = ["Carrier capacity", "Warehouse handoff", "Weather", "Peak demand", "Remote destination", "Other"]
    weights = np.array([0.30, 0.22, 0.14, 0.15, 0.09, 0.10], dtype=float)
    weights[2] += 0.18 * weather_severity
    weights[1] += max(warehouse_load_pct - 80.0, 0.0) / 100.0
    weights[3] += 0.20 if holiday_period else 0.0
    weights[4] += 0.18 if destination in ISLAND_DESTINATIONS else 0.0
    weights /= weights.sum()
    return str(rng.choice(labels, p=weights))


def generate_shipments(settings: SimulationSettings = SimulationSettings()) -> pd.DataFrame:
    """Generate labelled historical data without deterministic carrier outcomes."""
    if settings.rows < 100:
        raise ValueError("rows must be at least 100 so the simulated patterns are meaningful")

    rng = np.random.default_rng(settings.seed)
    calendar = pd.date_range(settings.start_date, settings.end_date, freq="D")
    ship_dates = pd.DatetimeIndex(rng.choice(calendar.to_numpy(), size=settings.rows, replace=True))

    origins = rng.choice(ORIGINS, size=settings.rows, p=ORIGIN_WEIGHTS)
    destinations = rng.choice(DESTINATIONS, size=settings.rows, p=DESTINATION_WEIGHTS)
    same_city = origins == destinations
    while same_city.any():
        destinations[same_city] = rng.choice(DESTINATIONS, size=int(same_city.sum()), p=DESTINATION_WEIGHTS)
        same_city = origins == destinations

    carriers = rng.choice(CARRIERS, size=settings.rows, p=CARRIER_WEIGHTS)
    service_levels = rng.choice(SERVICE_LEVELS, size=settings.rows, p=SERVICE_WEIGHTS)
    routes = np.char.add(np.char.add(origins.astype(str), "-"), destinations.astype(str))

    distance_km = np.array(
        [estimate_distance_km(str(origin), str(destination)) for origin, destination in zip(origins, destinations)],
        dtype=int,
    )
    package_weight_kg = np.clip(rng.lognormal(mean=1.25, sigma=0.85, size=settings.rows), 0.15, 45.0)
    volume_noise = rng.lognormal(mean=7.0, sigma=0.55, size=settings.rows)
    package_volume_cm3 = np.clip(package_weight_kg * 950.0 + volume_noise, 400.0, 85_000.0)
    order_value_eur = np.clip(rng.lognormal(mean=4.35, sigma=0.90, size=settings.rows), 8.0, 2_500.0)
    fragile = rng.random(settings.rows) < 0.13
    pickup_hour = np.clip(np.rint(rng.normal(13.5, 2.5, settings.rows)), 8, 19).astype(int)

    month = ship_dates.month.to_numpy()
    winter = np.isin(month, [1, 2, 11, 12])
    weather_severity = rng.choice([0, 1, 2, 3], size=settings.rows, p=[0.70, 0.22, 0.07, 0.01])
    weather_severity = np.clip(
        weather_severity + (winter & (rng.random(settings.rows) < 0.16)).astype(int), 0, 3
    )
    holiday_period = _holiday_period(ship_dates)
    warehouse_load_pct = np.clip(
        rng.normal(67.0, 10.0, settings.rows)
        + holiday_period * 15.0
        + np.isin(month, [11, 12]) * 5.0,
        35.0,
        100.0,
    )

    base_otp = np.array([CARRIER_BASE_ON_TIME_RATE[str(carrier)] for carrier in carriers])
    time_position = (ship_dates - calendar.min()).days.to_numpy() / max((calendar.max() - calendar.min()).days, 1)
    carrier_recent_on_time_rate = np.clip(
        base_otp
        - holiday_period * 0.035
        - (weather_severity >= 2) * 0.020
        + 0.012 * time_position
        + rng.normal(0.0, 0.018, settings.rows),
        0.72,
        0.98,
    )

    promised_days = np.array(
        [
            calculate_promised_days(str(origin), str(destination), str(service), str(carrier), int(distance))
            for origin, destination, service, carrier, distance in zip(
                origins, destinations, service_levels, carriers, distance_km
            )
        ],
        dtype=int,
    )

    cost_noise = rng.normal(0.0, 0.55, settings.rows)
    shipping_cost = np.array(
        [
            quote_shipping_cost(
                str(carrier),
                str(service),
                float(weight),
                float(volume),
                int(distance),
                bool(is_fragile),
                float(noise),
            )
            for carrier, service, weight, volume, distance, is_fragile, noise in zip(
                carriers,
                service_levels,
                package_weight_kg,
                package_volume_cm3,
                distance_km,
                fragile,
                cost_noise,
            )
        ]
    )

    carrier_effect = pd.Series(carriers).map(
        {"DHL": -0.38, "UPS": -0.28, "GLS": -0.02, "BRT": 0.16, "Poste Italiane": 0.36}
    ).to_numpy()
    service_effect = pd.Series(service_levels).map({"Express": -0.34, "Standard": 0.0, "Economy": 0.42}).to_numpy()
    island = np.isin(destinations, list(ISLAND_DESTINATIONS))
    south = np.isin(destinations, list(SOUTHERN_DESTINATIONS))
    late_pickup = pickup_hour >= 16
    heavy = np.maximum(package_weight_kg - 12.0, 0.0) / 12.0
    risk_logit = (
        -2.05
        + carrier_effect
        + service_effect
        + 0.00075 * distance_km
        + 0.034 * (warehouse_load_pct - 68.0)
        + 0.48 * weather_severity
        + 0.58 * holiday_period
        + 0.30 * late_pickup
        + 0.22 * heavy
        + 4.8 * (0.90 - carrier_recent_on_time_rate)
        + 0.30 * island
        + 0.12 * south
        + 0.20 * np.isin(month, [11, 12])
        + 0.24 * ((carriers == "Poste Italiane") & island)
        + 0.16 * ((carriers == "BRT") & south)
        - 0.12 * ((carriers == "DHL") & (distance_km < 350))
        + rng.normal(0.0, 0.30, settings.rows)
    )
    delay_probability = np.clip(_sigmoid(risk_logit), 0.015, 0.92)
    is_delayed = rng.random(settings.rows) < delay_probability
    delay_days = np.where(
        is_delayed,
        1
        + rng.poisson(
            0.55 + 0.28 * weather_severity + 0.25 * holiday_period + 0.18 * island,
            settings.rows,
        ),
        0,
    )
    delay_days = np.clip(delay_days, 0, 7).astype(int)
    early_days = rng.binomial(1, 0.28, settings.rows)
    transit_days = np.where(is_delayed, promised_days + delay_days, np.maximum(1, promised_days - early_days))
    delivery_dates = ship_dates + pd.to_timedelta(transit_days, unit="D")
    promised_delivery_dates = ship_dates + pd.to_timedelta(promised_days, unit="D")

    delay_reasons = [
        _choose_delay_reason(
            rng,
            bool(delayed),
            int(weather),
            float(load),
            bool(is_holiday),
            str(destination),
        )
        for delayed, weather, load, is_holiday, destination in zip(
            is_delayed, weather_severity, warehouse_load_pct, holiday_period, destinations
        )
    ]

    df = pd.DataFrame(
        {
            "ship_date": ship_dates,
            "promised_delivery_date": promised_delivery_dates,
            "delivery_date": delivery_dates,
            "carrier": carriers,
            "origin": origins,
            "destination": destinations,
            "route": routes,
            "service_level": service_levels,
            "package_weight_kg": np.round(package_weight_kg, 2),
            "package_volume_cm3": np.round(package_volume_cm3).astype(int),
            "order_value_eur": np.round(order_value_eur, 2),
            "fragile": fragile.astype(int),
            "pickup_hour": pickup_hour,
            "distance_km": distance_km,
            "warehouse_load_pct": np.round(warehouse_load_pct, 1),
            "carrier_recent_on_time_rate": np.round(carrier_recent_on_time_rate, 4),
            "weather_severity": weather_severity,
            "holiday_period": holiday_period.astype(int),
            "shipping_cost": shipping_cost,
            "promised_days": promised_days,
            "transit_days": transit_days.astype(int),
            "delay_days": delay_days,
            "is_delayed": is_delayed.astype(int),
            "delay_reason": delay_reasons,
            "status": "delivered",
            "data_source": "SIMULATED",
        }
    ).sort_values(["ship_date", "carrier", "route"], kind="stable")

    df.insert(0, "shipment_id", [f"SIM-{index:06d}" for index in range(1, len(df) + 1)])
    date_columns = ["ship_date", "promised_delivery_date", "delivery_date"]
    for column in date_columns:
        df[column] = pd.to_datetime(df[column]).dt.strftime("%Y-%m-%d")
    return df.reset_index(drop=True)


def data_dictionary() -> pd.DataFrame:
    """Return field definitions and whether each field is allowed at prediction time."""
    rows = [
        ("shipment_id", "string", "Unique simulated shipment identifier", "No"),
        ("ship_date", "date", "Dispatch date", "Yes"),
        ("promised_delivery_date", "date", "Contractual delivery date", "No; derived from promised_days"),
        ("delivery_date", "date", "Actual delivery date", "No; outcome leakage"),
        ("carrier", "category", "Selected logistics carrier", "Yes"),
        ("origin", "category", "Origin city", "Yes"),
        ("destination", "category", "Destination city", "Yes"),
        ("route", "category", "Origin-destination pair", "Yes"),
        ("service_level", "category", "Economy, Standard, or Express", "Yes"),
        ("package_weight_kg", "number", "Physical package weight in kilograms", "Yes"),
        ("package_volume_cm3", "number", "Package volume in cubic centimetres", "Yes"),
        ("order_value_eur", "number", "Declared order value in euros", "Yes"),
        ("fragile", "0/1", "Fragile handling indicator", "Yes"),
        ("pickup_hour", "integer", "Planned pickup hour", "Yes"),
        ("distance_km", "integer", "Estimated route distance", "Yes"),
        ("warehouse_load_pct", "number", "Warehouse utilisation when dispatched", "Yes"),
        ("carrier_recent_on_time_rate", "rate", "Carrier trailing on-time performance available at dispatch", "Yes"),
        ("weather_severity", "0-3", "Forecast disruption severity", "Yes"),
        ("holiday_period", "0/1", "Peak holiday-period indicator", "Yes"),
        ("shipping_cost", "EUR", "Quoted shipping cost", "Yes"),
        ("promised_days", "integer", "Promised transit time", "Yes"),
        ("transit_days", "integer", "Actual transit time", "No; outcome"),
        ("delay_days", "integer", "Days delivered after promise", "No; outcome"),
        ("is_delayed", "0/1", "Training label: actual delivery later than promise", "Target"),
        ("delay_reason", "category", "Observed reason when delayed", "No; outcome"),
        ("status", "category", "Final shipment status", "No; outcome"),
        ("data_source", "string", "Explicit simulated-data marker", "No"),
    ]
    return pd.DataFrame(rows, columns=["column", "type", "description", "prediction_use"])


def save_simulated_data(
    output_csv: str | Path,
    dictionary_csv: str | Path,
    settings: SimulationSettings = SimulationSettings(),
) -> pd.DataFrame:
    output_csv = Path(output_csv)
    dictionary_csv = Path(dictionary_csv)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    dictionary_csv.parent.mkdir(parents=True, exist_ok=True)
    df = generate_shipments(settings)
    df.to_csv(output_csv, index=False)
    data_dictionary().to_csv(dictionary_csv, index=False)
    return df
