"""Shared constants for simulation, training, and prediction."""

from __future__ import annotations


RANDOM_SEED = 42

CITY_COORDINATES = {
    "Milan": (45.4642, 9.1900),
    "Rome": (41.9028, 12.4964),
    "Naples": (40.8518, 14.2681),
    "Turin": (45.0703, 7.6869),
    "Florence": (43.7696, 11.2558),
    "Bologna": (44.4949, 11.3426),
    "Bari": (41.1171, 16.8719),
    "Palermo": (38.1157, 13.3615),
    "Cagliari": (39.2238, 9.1217),
    "Genoa": (44.4056, 8.9463),
    "Venice": (45.4408, 12.3155),
    "Verona": (45.4384, 10.9916),
}

ORIGINS = ["Milan", "Rome", "Bologna", "Turin", "Verona"]
ORIGIN_WEIGHTS = [0.42, 0.22, 0.16, 0.12, 0.08]

DESTINATIONS = list(CITY_COORDINATES)
DESTINATION_WEIGHTS = [
    0.11,  # Milan
    0.16,  # Rome
    0.12,  # Naples
    0.09,  # Turin
    0.09,  # Florence
    0.10,  # Bologna
    0.08,  # Bari
    0.06,  # Palermo
    0.05,  # Cagliari
    0.07,  # Genoa
    0.04,  # Venice
    0.03,  # Verona
]

CARRIERS = ["DHL", "UPS", "GLS", "BRT", "Poste Italiane"]
CARRIER_WEIGHTS = [0.22, 0.20, 0.20, 0.23, 0.15]

SERVICE_LEVELS = ["Economy", "Standard", "Express"]
SERVICE_WEIGHTS = [0.18, 0.62, 0.20]

CARRIER_BASE_ON_TIME_RATE = {
    "DHL": 0.935,
    "UPS": 0.925,
    "GLS": 0.895,
    "BRT": 0.875,
    "Poste Italiane": 0.845,
}

CARRIER_BASE_COST = {
    "DHL": 7.40,
    "UPS": 7.10,
    "GLS": 6.10,
    "BRT": 5.80,
    "Poste Italiane": 5.10,
}

SERVICE_COST_MULTIPLIER = {
    "Economy": 0.86,
    "Standard": 1.00,
    "Express": 1.52,
}

ISLAND_DESTINATIONS = {"Palermo", "Cagliari"}
SOUTHERN_DESTINATIONS = {"Naples", "Bari", "Palermo", "Cagliari"}

