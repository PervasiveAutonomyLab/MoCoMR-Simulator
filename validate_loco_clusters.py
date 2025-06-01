import pandas as pd
import numpy as np
import re
import matplotlib.pyplot as plt
from sklearn.mixture import GaussianMixture
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score

from datetime import datetime, timedelta
import ast

# === Load CSV ===
loc_df = pd.read_csv("loc_gmm_clustering_results.csv", low_memory=False)
loc_df["Timestamps"] = loc_df["Timestamps"].apply(ast.literal_eval)

# === Raw Duration Extractor ===
def get_location_durations(cluster_idx):
    diffs = []
    cluster_df = loc_df[loc_df["Cluster_GMM"] == cluster_idx]
    for _, row in cluster_df.iterrows():
        ts = row["Timestamps"]
        diffs += [ts[i] - ts[i-1] for i in range(1, len(ts))]
    return np.array(diffs)

# === Generator using KDE ===
def generate_location_log(cluster_idx):
    try:
        df = pd.read_csv("loc_gmm_clustering_results.csv", low_memory=False)
        df["Timestamps"] = df["Timestamps"].apply(ast.literal_eval)
        df["X"] = df["X"].apply(ast.literal_eval)
        df["Y"] = df["Y"].apply(ast.literal_eval)
        df["Z"] = df["Z"].apply(ast.literal_eval)

        cluster_df = df[df["Cluster_GMM"] == cluster_idx]
        if cluster_df.empty:
            return [f"No movement data available for cluster {cluster_idx}."]

        # Model delta values (dt, dx, dy, dz)
        deltas = []
        for _, row in cluster_df.iterrows():
            ts, xs, ys, zs = row["Timestamps"], row["X"], row["Y"], row["Z"]
            for i in range(1, len(ts)):
                dt = ts[i] - ts[i-1]
                dx = xs[i] - xs[i-1]
                dy = ys[i] - ys[i-1]
                dz = zs[i] - zs[i-1]
                deltas.append([dt, dx, dy, dz])

        deltas = np.array(deltas)
        if deltas.shape[0] < 2:
            return [f"Not enough data to model movement for cluster {cluster_idx}."]
        
        mean = np.mean(deltas, axis=0)
        cov = np.cov(deltas.T)

        samples = np.random.multivariate_normal(mean, cov,  size=len(deltas))
        base_time = datetime.now()
        t = 0
        x, y, z = 0.0, 0.0, 0.0
        logs = []

        for dt, dx, dy, dz in samples:
            t += max(dt, 0.01)
            x += dx
            y += dy
            z += dz
            time_str = (base_time + timedelta(seconds=t)).strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]
            logs.append(f"{time_str}: Position - X: {x:.2f}, Y: {y:.2f}, Z: {z:.2f} | Duration: {max(dt,0):.2f}s")

        return logs

    except Exception as e:
        return [f"Error generating location log: {e}"]

# === Duration Extractor ===
def extract_durations(logs, pattern):
    return np.array([float(m.group(1)) for line in logs if (m := re.search(pattern, line))])

# === Clustering + Comparison ===
def compare_clusters(raw, gen, name="Modality"):
    raw = raw.reshape(-1, 1)
    gen = gen.reshape(-1, 1)

    raw_labels = GaussianMixture(n_components=2).fit_predict(raw)
    gen_labels = GaussianMixture(n_components=2).fit_predict(gen)

    min_len = min(len(raw_labels), len(gen_labels))
    ari = adjusted_rand_score(raw_labels[:min_len], gen_labels[:min_len])
    nmi = normalized_mutual_info_score(raw_labels[:min_len], gen_labels[:min_len])

    print(f"--- {name} ---")
    print(f"Adjusted Rand Index: {ari:.4f}")
    print(f"Normalized Mutual Info: {nmi:.4f}\\n")

    plt.hist(raw, bins=30, alpha=0.5, label="Raw", density=True)
    plt.hist(gen, bins=30, alpha=0.5, label="Generated", density=True)
    plt.legend()
    plt.title(f"{name} Duration Distribution")
    plt.xlabel("Duration (s)")
    plt.show()

# === Run it! ===
try:
    cluster_idx = int(input("Enter cluster number (1-based index): ")) - 1
    loc_raw = get_location_durations(cluster_idx)
    loc_gen = extract_durations(generate_location_log(cluster_idx), r'([\d\.]+)(?=s\b)')
    compare_clusters(loc_raw, loc_gen, name="Locomotion")
except Exception as e:
    print("Error:", e)
