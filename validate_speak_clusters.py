# validate_speaking_clusters.py

import pandas as pd
import numpy as np
import re
import ast
import matplotlib.pyplot as plt
from sklearn.mixture import GaussianMixture
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score

from scipy.stats import gaussian_kde
from datetime import datetime, timedelta

# === Load CSV ===
speak_df = pd.read_csv("gmm_clustering_results_speaking.csv")

# === Raw Duration Extractor ===
def get_speaking_durations(cluster_idx):
    df = pd.read_csv("gmm_clustering_results_speaking.csv")
    df["Start Times"] = df["Start Times"].apply(ast.literal_eval)
    df["Durations"] = df["Durations"].apply(ast.literal_eval)

    cluster_df = df[df["Cluster_GMM"] == cluster_idx]
    durations = []
    for _, row in cluster_df.iterrows():
        durations += [d for d in row["Durations"] if 0.01 < d < 10]

    return np.array(durations)

# === Log Generator ===
def generate_speaking_log(cluster_idx):
    try:
        speak_df["Start Times"] = speak_df["Start Times"].apply(ast.literal_eval)
        speak_df["Durations"] = speak_df["Durations"].apply(ast.literal_eval)
        cluster_df = speak_df[speak_df["Cluster_GMM"] == cluster_idx]

        all_starts, all_durs = [], []
        for _, row in cluster_df.iterrows():
            all_starts += row["Start Times"]
            all_durs += row["Durations"]

        if not all_starts or not all_durs:
            return [f"No valid speaking data for cluster {cluster_idx}."]

        # FFT smoothing for start times (unchanged)
        start_hist, edges = np.histogram(all_starts, bins='auto', density=True)
        start_smoothed = np.fft.irfft(np.fft.rfft(start_hist)[:4], n=len(start_hist))
        start_probs = np.maximum(start_smoothed, 0)
        start_probs /= start_probs.sum()
        start_bins = (edges[:-1] + edges[1:]) / 2
        sampled_starts = np.random.choice(start_bins, size=len(all_starts), p=start_probs)

        # KDE smoothing for durations (NEW)
        durs = np.array(all_durs)
        durs = durs[(durs > 0.01) & (durs < 10)]
        if len(durs) < 10:
            return [f"Not enough valid durations in cluster {cluster_idx}."]

        from scipy.stats import gaussian_kde
        xs = np.linspace(0.01, 10.0, 500)
        kde = gaussian_kde(durs)
        probs = kde(xs)
        probs = np.maximum(probs, 0)
        probs /= probs.sum()

        sampled_durs = np.random.choice(xs, size=len(durs), p=probs)

        # Generate logs
        base_time = datetime.now()
        logs = []
        for ts, dur in zip(sampled_starts, sampled_durs):
            time_str = (base_time + timedelta(seconds=ts)).strftime("%Y-%m-%d %H:%M:%S")
            logs.append(f"{time_str}: Speaking Event lasted {dur:.2f} seconds")
        return logs

    except Exception as e:
        return [f"Error generating speaking log: {e}"]



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
    print(f"Normalized Mutual Info: {nmi:.4f}\n")

    plt.hist(raw, bins=30, alpha=0.5, label="Raw", density=True)
    plt.hist(gen, bins=30, alpha=0.5, label="Generated", density=True)
    plt.legend()
    plt.title(f"{name} Duration Distribution")
    plt.xlabel("Duration (s)")
    plt.show()

# === Run it! ===
try:
    cluster_idx = int(input("Enter cluster number (1-based index): ")) - 1

    speak_raw = get_speaking_durations(cluster_idx)
    logs = generate_speaking_log(cluster_idx)
    if any("Error" in line or "No valid" in line for line in logs):
        print(logs[0])
    else:
        speak_gen = extract_durations(logs, r'([\d\.]+)(?= seconds\b)')
        if len(speak_raw) < 2 or len(speak_gen) < 2:
            print(f"Not enough data to compare clusters for cluster {cluster_idx}.")
        else:
            compare_clusters(speak_raw, speak_gen, name="Speaking")

except Exception as e:
    print("Error:", e)
