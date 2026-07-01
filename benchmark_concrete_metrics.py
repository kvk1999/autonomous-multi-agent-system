import time
import numpy as np
import torch

# Benchmark script for "Concrete Metrics".
# Generates synthetic lat/lon points and measures distance-matrix compute time
# for CPU (nested-loop baseline) vs GPU (PyTorch tensor broadcasting).


def haversine_distance_cpu_naive(lats, lons):
    n = len(lats)
    dist_matrix = np.zeros((n, n), dtype=np.float64)
    R = 6371.0

    for i in range(n):
        lat1 = np.radians(lats[i])
        lon1 = np.radians(lons[i])
        for j in range(n):
            if i == j:
                continue
            lat2 = np.radians(lats[j])
            lon2 = np.radians(lons[j])

            dlat = lat2 - lat1
            dlon = lon2 - lon1
            a = (np.sin(dlat / 2.0) ** 2) + np.cos(lat1) * np.cos(lat2) * (np.sin(dlon / 2.0) ** 2)
            c = 2.0 * np.arcsin(np.sqrt(a))
            dist_matrix[i, j] = R * c
    return dist_matrix


def haversine_distance_gpu(lats, lons, use_cuda=True):
    device = torch.device("cuda" if (use_cuda and torch.cuda.is_available()) else "cpu")

    lats_t = torch.tensor(lats, dtype=torch.float32, device=device)
    lons_t = torch.tensor(lons, dtype=torch.float32, device=device)

    lats_rad = torch.deg2rad(lats_t)
    lons_rad = torch.deg2rad(lons_t)

    lat1 = lats_rad.unsqueeze(1)
    lat2 = lats_rad.unsqueeze(0)
    lon1 = lons_rad.unsqueeze(1)
    lon2 = lons_rad.unsqueeze(0)

    dlat = lat2 - lat1
    dlon = lon2 - lon1

    a = torch.sin(dlat / 2.0) ** 2 + torch.cos(lat1) * torch.cos(lat2) * torch.sin(dlon / 2.0) ** 2
    c = 2.0 * torch.asin(torch.sqrt(a))

    R = 6371.0
    dist_matrix = R * c
    return dist_matrix


def generate_synthetic_points(n: int, seed: int = 1234):
    # Synthetic points roughly clustered around APAC to look realistic.
    # Mix a few hub regions so Zone logic feels grounded.
    rng = np.random.default_rng(seed)

    hubs = [
        (19.0760, 72.8777),   # Mumbai
        (-6.2088, 106.8456),  # Jakarta
        (1.3521, 103.8198),   # Singapore
        (14.5995, 120.9842), # Manila
        (13.0827, 80.2707),  # Chennai
        (22.5726, 88.3639),  # Kolkata
    ]
    hub_idx = rng.integers(0, len(hubs), size=n)

    # Add small offsets in degrees.
    lat_offsets = rng.normal(0, 0.35, size=n)
    lon_offsets = rng.normal(0, 0.45, size=n)

    base_lats = np.array([hubs[i][0] for i in hub_idx], dtype=np.float64)
    base_lons = np.array([hubs[i][1] for i in hub_idx], dtype=np.float64)

    lats = base_lats + lat_offsets
    lons = base_lons + lon_offsets
    return lats, lons


def time_it(fn, *args, **kwargs):
    start = time.perf_counter()
    out = fn(*args, **kwargs)
    # Ensure GPU work is complete before timing ends.
    if torch.is_tensor(out) and out.is_cuda:
        torch.cuda.synchronize()
    return out, (time.perf_counter() - start)


def main():
    # Sizes chosen to match presentation requirements.
    n_points_list = [1000, 2000, 5000, 10000]

    print("=== Concrete Metrics Benchmark ===")
    print(f"CUDA available: {torch.cuda.is_available()}")

    results = []
    for n in n_points_list:
        print(f"\n--- N = {n} ---")
        lats, lons = generate_synthetic_points(n)

        # CPU timing: nested-loop baseline can get slow; keep N <= 2000 for CPU if needed.
        if n <= 2000:
            _, cpu_time = time_it(haversine_distance_cpu_naive, lats, lons)
            print(f"CPU naive time: {cpu_time:.4f}s")
        else:
            cpu_time = None
            print("CPU naive time: skipped (size too large for CPU baseline) ")

        # GPU timing
        # Warmup small call
        _ = haversine_distance_gpu(lats[:10], lons[:10], use_cuda=True)
        if torch.cuda.is_available():
            torch.cuda.synchronize()

        _, gpu_time = time_it(haversine_distance_gpu, lats, lons, True)
        print(f"GPU torch time: {gpu_time:.4f}s")

        results.append({"size": n, "cpu_time": cpu_time, "gpu_time": gpu_time})

    print("\n=== RAW RESULTS (copy/paste) ===")
    for r in results:
        print(r)


if __name__ == "__main__":
    main()

