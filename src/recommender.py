"""Transport recommendation via weighted multi-criteria scoring.

Reference values below are REAL, cited figures (web research, Sept 2026) for Indian
urban conditions -- not invented placeholders. They are still national/representative
averages, not measurements for a specific city -- state this in your report.

Speed (km/h), by congestion level:
  - Car/Carpool: 30 (light) / 22 (national urban avg, Global Traffic Index 2025) /
    14 (Bengaluru peak-hour avg, Global Traffic Index 2025)
  - Bus: 25 / 18 / 9 (Mumbai peak-hour bus avg, public bus transit literature)
  - Metro: 40 flat -- dedicated track, unaffected by road congestion
    (Delhi Metro operational avg incl. stops, DMRC/Wikipedia)

Cost:
  - Car: Rs 6/km (petrol, solo occupant; Rs 5-7/km at ~Rs102/L, 15-20 km/L -- fuel
    cost calculator aggregators, 2026)
  - Bus: Rs 1.5/km (government bus fares range Rs 1-3/km -- Haryana/UP Roadways
    official fare notifications)
  - Metro: real DMRC distance-based fare slabs (not a flat rate)
  - Carpool: Car cost split across an assumed 2 occupants

Emissions (g CO2 per passenger-km):
  - Car: 157.5 (India corporate average fleet figure, 2021-22)
  - Bus: 53.1 (20-seat CNG bus, already a per-passenger figure)
  - Metro: 25 (Mumbai Metro underground lifecycle assessment)
  - Carpool: Car's per-vehicle figure split across 2 occupants
"""

SPEED_KMPH = {
    "Car":     {"Low": 30, "Medium": 22, "High": 14},
    "Bus":     {"Low": 25, "Medium": 18, "High": 9},
    "Metro":   {"Low": 40, "Medium": 40, "High": 40},
    "Carpool": {"Low": 30, "Medium": 22, "High": 14},
}

COST_PER_KM = {
    "Car": 6.0,
    "Bus": 1.5,
    "Carpool": 3.0,  # Car's Rs 6/km split across 2 occupants
}

# Real DMRC (Delhi Metro) distance-based fare slabs, 2026
METRO_FARE_SLABS = [
    (2, 10), (5, 20), (12, 30), (21, 40), (32, 50), (float("inf"), 60),
]

EMISSION_PER_KM = {
    "Car": 157.5,
    "Bus": 53.1,
    "Metro": 25.0,
    "Carpool": 78.75,  # Car's 157.5 g/km split across 2 occupants
}

# Fixed access/waiting overhead (minutes) -- walking to a stop/station, waiting for
# a vehicle, parking, coordinating a pickup. Standard transport-planning assumption
# (access/egress time), not precisely sourced for a specific city. Without this,
# mass transit unrealistically dominates every scenario, since it has no "door to
# door" penalty otherwise.
ACCESS_TIME_MIN = {
    "Car": 2,
    "Bus": 8,
    "Metro": 12,
    "Carpool": 5,
}

DEFAULT_WEIGHTS = {"time": 0.4, "cost": 0.3, "environment": 0.3}


def metro_fare(distance_km: float) -> float:
    for max_km, fare in METRO_FARE_SLABS:
        if distance_km <= max_km:
            return fare
    return METRO_FARE_SLABS[-1][1]


def compute_mode_metrics(distance_km: float, congestion_level: str):
    """Return raw time (min), cost (Rs), and emissions (g CO2) for each mode."""
    metrics = {}
    for mode in SPEED_KMPH:
        speed = SPEED_KMPH[mode][congestion_level]
        time_min = (distance_km / speed) * 60 + ACCESS_TIME_MIN[mode]

        cost = metro_fare(distance_km) if mode == "Metro" else distance_km * COST_PER_KM[mode]

        metrics[mode] = {
            "time_min": time_min,
            "cost": cost,
            "emission_g": distance_km * EMISSION_PER_KM[mode],
        }
    return metrics


def _normalize(values: dict) -> dict:
    """Min-max normalize a {mode: value} dict to 0 (best/lowest) - 1 (worst/highest)."""
    vmin, vmax = min(values.values()), max(values.values())
    if vmax == vmin:
        return {k: 0.0 for k in values}
    return {k: (v - vmin) / (vmax - vmin) for k, v in values.items()}


def recommend_transport(distance_km: float, congestion_level: str, weights: dict = None):
    """Rank all transport modes by weighted score (lower = better).

    Returns a list of (mode, score, breakdown) sorted best-first.
    """
    weights = weights or DEFAULT_WEIGHTS
    metrics = compute_mode_metrics(distance_km, congestion_level)

    norm_time = _normalize({m: v["time_min"] for m, v in metrics.items()})
    norm_cost = _normalize({m: v["cost"] for m, v in metrics.items()})
    norm_env = _normalize({m: v["emission_g"] for m, v in metrics.items()})

    ranked = []
    for mode in SPEED_KMPH:
        score = (
            weights["time"] * norm_time[mode]
            + weights["cost"] * norm_cost[mode]
            + weights["environment"] * norm_env[mode]
        )
        breakdown = {
            "time_min": round(metrics[mode]["time_min"], 1),
            "cost": round(metrics[mode]["cost"], 1),
            "emission_g": round(metrics[mode]["emission_g"], 1),
            "time_score": round(norm_time[mode], 2),
            "cost_score": round(norm_cost[mode], 2),
            "environment_score": round(norm_env[mode], 2),
        }
        ranked.append((mode, round(score, 3), breakdown))

    ranked.sort(key=lambda x: x[1])
    return ranked


if __name__ == "__main__":
    scenarios = [
        ("Short trip, Low congestion", 5, "Low"),
        ("Medium trip, High congestion", 15, "High"),
        ("Long trip, Medium congestion, eco-focused", 30, "Medium"),
    ]

    for label, distance, congestion in scenarios:
        print(f"\n=== {label} (distance={distance}km, congestion={congestion}) ===")
        weights = DEFAULT_WEIGHTS
        if "eco-focused" in label:
            weights = {"time": 0.2, "cost": 0.2, "environment": 0.6}
        ranked = recommend_transport(distance, congestion, weights)
        for mode, score, breakdown in ranked:
            print(f"{mode}: score={score}  {breakdown}")
        print(f"--> Recommended: {ranked[0][0]}")
