import os
import csv
import numpy as np

# Generates mock APAC delivery dataset for local/offline demos.
# Output: mock_apac_deliveries.csv in this project folder.

OUTPUT_FILE = "mock_apac_deliveries.csv"


def main(rows: int = 10000, seed: int = 2025):
    rng = np.random.default_rng(seed)

    hubs = [
        ("Mumbai_Zone_A", 19.0760, 72.8777),   # India
        ("Chennai_Zone_B", 13.0827, 80.2707), # India
        ("Kolkata_Zone_C", 22.5726, 88.3639),# India
        ("Jakarta_Port_B", -6.2088, 106.8456),# Indonesia
        ("Singapore_Port_A", 1.3521, 103.8198),# Singapore
        ("Manila_Zone_D", 14.5995, 120.9842),# Philippines
        ("Bangkok_Zone_E", 13.7563, 100.5018),# Thailand
        ("Colombo_Zone_F", 6.9271, 79.8612),  # Sri Lanka
    ]

    statuses = ["Pending", "In_Transit", "Delivered", "Exception"]
    status_probs = [0.35, 0.45, 0.18, 0.02]

    # Lat/lon jitter scales (degrees)
    lat_jitter = 0.35
    lon_jitter = 0.45

    with open(OUTPUT_FILE, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["order_id", "latitude", "longitude", "city_zone", "status"])

        for i in range(rows):
            order_id = f"ORD-{10000 + i}"
            hub = hubs[int(rng.integers(0, len(hubs)))]
            zone_name, hub_lat, hub_lon = hub

            lat = hub_lat + rng.normal(0, lat_jitter)
            lon = hub_lon + rng.normal(0, lon_jitter)

            status = statuses[int(np.searchsorted(np.cumsum(status_probs), rng.random()))]

            writer.writerow([order_id, f"{lat:.6f}", f"{lon:.6f}", zone_name, status])

    print(f"Created {OUTPUT_FILE} with {rows} rows")


if __name__ == "__main__":
    main()

