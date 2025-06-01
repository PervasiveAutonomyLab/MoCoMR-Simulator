# validate_clusters.py

import pandas as pd
import numpy as np
import re
import ast
import matplotlib.pyplot as plt
from sklearn.mixture import GaussianMixture
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score

from scipy.stats import gaussian_kde
from datetime import datetime, timedelta

# === Load CSVs ===
gaze_cluster_df = pd.read_csv("hover_gmm_user_clustering_results.csv")
gaze_fft_df = pd.read_csv("hover_user_object_fft.csv")

# === Raw Duration Extractors ===

def get_gaze_durations(cluster_idx):
    gaze_cluster_df["Group ID"] = gaze_cluster_df["Group ID"].astype(str)
    gaze_cluster_df["Cluster_GMM"] = pd.to_numeric(gaze_cluster_df["Cluster_GMM"], errors="coerce")
    gids = gaze_cluster_df[gaze_cluster_df["Cluster_GMM"] == cluster_idx]["Group ID"].unique()
    gaze_fft_df["Group ID"] = gaze_fft_df["Group ID"].astype(str)
    return np.array(gaze_fft_df[gaze_fft_df["Group ID"].isin(gids)]["Mean Duration"].tolist())

# === Log Generators (from your code) ===
def generate_gaze_log(cluster_idx):
    try:
        # Load cluster info
        cluster_df = pd.read_csv("hover_gmm_user_clustering_results.csv")
        cluster_df["Group ID"] = cluster_df["Group ID"].astype(str)
        cluster_df["Cluster_GMM"] = pd.to_numeric(cluster_df["Cluster_GMM"], errors="coerce")
        group_ids = cluster_df[cluster_df["Cluster_GMM"] == cluster_idx]["Group ID"].unique()

        if len(group_ids) == 0:
            return [f"No gaze groups found for cluster {cluster_idx}."]

        # Load FFT durations
        fft_df = pd.read_csv("hover_user_object_fft.csv")
        fft_df["Group ID"] = fft_df["Group ID"].astype(str)
        filtered = fft_df[fft_df["Group ID"].isin(group_ids)]
        durations = filtered["Mean Duration"].dropna().to_numpy()

        # Hard limit to match true data distribution shape
        durations = durations[(durations > 0.01) & (durations < 0.6)]
        if len(durations) < 20:
            return [f"Not enough valid gaze durations for cluster {cluster_idx}."]

        # Build KDE model
        kde = gaussian_kde(durations)
        xs = np.linspace(0.01, 0.6, 500)
        probs = kde(xs)
        probs = np.maximum(probs, 0)
        probs /= probs.sum()

        # Sample from KDE
        sampled_durations = np.random.choice(xs, size=len(durations), p=probs)

        # Format into logs with timestamps
        base_time = datetime.now()
        logs = []
        for i, dur in enumerate(sampled_durations):
            time_str = (base_time + timedelta(seconds=i)).strftime("%Y-%m-%d %H:%M:%S")
            logs.append(f"{time_str}: Gazed at object for {dur:.2f}s")
        return logs

    except Exception as e:
        return [f"Error generating gaze log: {e}"]

# === Duration Extractor from Logs ===
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

    gaze_raw = get_gaze_durations(cluster_idx)
    gaze_gen = extract_durations(generate_gaze_log(cluster_idx), r'([\d\.]+)(?=s\b)')
    compare_clusters(gaze_raw, gaze_gen, name="Gaze")

except Exception as e:
    print("Error:", e)
