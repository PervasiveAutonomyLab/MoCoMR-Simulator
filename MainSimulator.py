import tkinter as tk
from tkinter import ttk
import pandas as pd
import numpy as np
import ast
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import sys
import re
import networkx as nx
from numpy.random import multivariate_normal
from scipy.stats import gaussian_kde
from matplotlib.lines import Line2D
from matplotlib.patches import FancyArrowPatch
import itertools
from networkx.algorithms import isomorphism


last_sim_logs = []


def on_closing():
    # stop the tk mainloop
    root.quit()
    # destroy any remaining windows
    root.destroy()
    # exit the Python process
    sys.exit(0)

def get_location_durations(cluster_idx):
    df = pd.read_csv("loc_gmm_clustering_results.csv", low_memory=False)
    df["Timestamps"] = df["Timestamps"].apply(ast.literal_eval)
    diffs = []
    for _, row in df[df["Cluster_GMM"] == cluster_idx].iterrows():
        ts = row["Timestamps"]
        diffs += [ts[i] - ts[i-1] for i in range(1, len(ts))]
    return np.array(diffs)

def get_speaking_durations(cluster_idx):
    df = pd.read_csv("gmm_clustering_results_speaking.csv")
    df["Start Times"] = df["Start Times"].apply(ast.literal_eval)
    df["Durations"]   = df["Durations"].apply(ast.literal_eval)
    durs = []
    for _, row in df[df["Cluster_GMM"] == cluster_idx].iterrows():
        durs += row["Durations"]
    return np.array(durs)

def get_gaze_durations(cluster_idx):
    cluster_df = pd.read_csv("hover_gmm_user_clustering_results.csv")
    cluster_df["Group ID"]     = cluster_df["Group ID"].astype(str)
    cluster_df["Cluster_GMM"]  = pd.to_numeric(cluster_df["Cluster_GMM"], errors="coerce")
    gids = cluster_df[cluster_df["Cluster_GMM"] == cluster_idx]["Group ID"].unique().tolist()
    fft_df = pd.read_csv("hover_user_object_fft.csv")
    fft_df["Group ID"] = fft_df["Group ID"].astype(str)
    return np.array(
        fft_df[fft_df["Group ID"].isin(gids)]["Mean Duration"].tolist()
    )

def show_histograms_raw():
    # pull cluster choices for all 4 participants
    loc_idxs   = [int(dd.get().split()[0]) - 1 for _, dd, _ in participant_dropdowns]
    speak_idxs = [int(dd.get().split()[0]) - 1 for _, _, dd in participant_dropdowns]
    gaze_idxs  = [int(dd.get().split()[0]) - 1 for dd, _, _ in participant_dropdowns]

    # 3 modalities × 4 participants
    fig, axes = plt.subplots(
        nrows=3, ncols=4,
        figsize=(8, 6),
        tight_layout=True
    )

    # (title, extractor, chosen cluster idxs)
    modalities = [
        ("Gaze Durations (s)",     get_gaze_durations,      gaze_idxs),
        ("Locomotion Δt (s)",      get_location_durations,  loc_idxs),
        ("Speaking Durations (s)", get_speaking_durations,  speak_idxs),
    ]
    # just the short row‐labels we want
    row_labels = ["Gaze", "Locomotion", "Speaking"]

    for row, (title, extractor, idxs) in enumerate(modalities):
        for col, cluster_idx in enumerate(idxs):
            ax = axes[row, col]
            data = extractor(cluster_idx)
            if data.size:
                ax.hist(data, bins=20)

            # only label the leftmost column with both row name & count
            if col == 0:
                ax.set_ylabel(f"{row_labels[row]}\nCount")
            # only label the bottom row with the x‐axis
            if row == 2:
                ax.set_xlabel("Duration (s)")
            # put participant title on the top row
            if row == 0:
                ax.set_title(f"P{col+1}")

    # overall figure title
    fig.suptitle("raw Histograms", y=1.02, fontsize=14)

    # embed into Tk window
    win = tk.Toplevel(root)
    win.title("raw Histograms")
    canvas = FigureCanvasTkAgg(fig, master=win)
    canvas.draw()
    canvas.get_tk_widget().pack(fill="both", expand=True)

def show_histograms_user():
    # pull cluster choices for all 4 participants
    loc_idxs   = [int(dd.get().split()[0]) - 1 for _, dd, _ in participant_dropdowns]
    speak_idxs = [int(dd.get().split()[0]) - 1 for _, _, dd in participant_dropdowns]
    gaze_idxs  = [int(dd.get().split()[0]) - 1 for dd, _, _ in participant_dropdowns]

    fig, axes = plt.subplots(3, 4, figsize=(8, 6), tight_layout=True)

    # now pair each row with the *generator* instead of the CSV‐getter
    modalities = [
        ("Gaze Durations (s)",     generate_gaze_log,     gaze_idxs),
        ("Locomotion Δt (s)",      generate_location_log, loc_idxs),
        ("Speaking Durations (s)", generate_speaking_log, speak_idxs),
    ]
    row_labels = ["Gaze", "Locomotion", "Speaking"]

    for row, (title, generator, idxs) in enumerate(modalities):
        for col, cluster_idx in enumerate(idxs):
            ax = axes[row, col]

            # 1) get the raw log‐lines
            logs = generator(cluster_idx)

            # 2) pull out any float immediately before 's' (for gaze & loco) or ' seconds'
            durations = []
            for line in logs:
                m = re.search(r'([\d\.]+)(?=s\b)', line) or \
                    re.search(r'([\d\.]+)(?= seconds\b)', line)
                if m:
                    durations.append(float(m.group(1)))

            # 3) plot exactly as before
            if durations:
                ax.hist(durations, bins=20)

            # axis‐label logic unchanged
            if col == 0:
                ax.set_ylabel(f"{row_labels[row]}\nCount")
            if row == 2:
                ax.set_xlabel("Duration (s)")
            if row == 0:
                ax.set_title(f"P{col+1}")

    fig.suptitle("Participant Histograms", y=1.02, fontsize=14)

    win = tk.Toplevel(root)
    win.title("Participant Histograms")
    canvas = FigureCanvasTkAgg(fig, master=win)
    canvas.draw()
    canvas.get_tk_widget().pack(fill="both", expand=True)

def parse_intervals(logs, pattern):
    """
    Parses a list of log strings to extract time intervals.
    `pattern` should have two groups: timestamp string and duration in seconds.
    Returns a list of (start_datetime, end_datetime) tuples.
    """
    intervals = []
    for line in logs:
        match = re.match(pattern, line)
        if match:
            ts_str, dur_str = match.groups()
            try:
                start = datetime.strptime(ts_str, "%Y-%m-%d %H:%M:%S")
            except ValueError:
                start = datetime.strptime(ts_str, "%Y-%m-%d %H:%M:%S.%f")
            duration = float(dur_str)
            end = start + timedelta(seconds=duration)
            intervals.append((start, end))
    return intervals

def show_sociograms():
    # 1) Get selected clusters
    #loc_idxs   = [int(dd.get().split()[0]) - 1 for _, dd, _ in participant_dropdowns]
    #speak_idxs = [int(dd.get().split()[0]) - 1 for _, _, dd in participant_dropdowns]
    #gaze_idxs  = [int(dd.get().split()[0]) - 1 for dd, _, _ in participant_dropdowns]

    if not last_sim_logs:
        tk.messagebox.showwarning("No data", "Please run the simulation first.")
        return

    # We ignore cluster‐dropdowns here since we cache per‐participant
    gaze_logs_list  = [p["gaze"]  for p in last_sim_logs]
    loco_logs_list  = [p["loc"]   for p in last_sim_logs]
    speak_logs_list = [p["speak"] for p in last_sim_logs]

    # 2) Helpers

    def overlap(a, b):
        total = 0
        for s1,e1 in a:
            for s2,e2 in b:
                dt = (min(e1,e2) - max(s1,s2)).total_seconds()
                if dt>0: total += dt
        return total

    # 3) Fixed node positions
    pos = {"P1":(0,1),"P2":(1,1),"P3":(0,0),"P4":(1,0)}

    # 4) Build the three graphs
    graphs, colors, titles = [], [], []

    # Gaze (undirected)
    G = nx.Graph(); G.add_nodes_from(pos)
    gaze_events = [
      parse_intervals(gaze_logs_list[i], r"(.+?): Gazed at object for ([\d\.]+)s")
      for i in range(4)
    ]
    for i in range(4):
        for j in range(i+1,4):
            w = overlap(gaze_events[i], gaze_events[j])
            if w>0: G.add_edge(f"P{i+1}",f"P{j+1}", weight=w)
    graphs.append(G); colors.append("blue"); titles.append("Gaze Sociogram")

    # Locomotion undirected)
    G = nx.Graph(); G.add_nodes_from(pos)
    loco_events = []
    loc_pat = r"(.+?): Position - X: ([\d\.]+), Y: ([\d\.]+), Z: ([\d\.]+) \| Duration: ([\d\.]+)s"
    for i in range(4):
        evs=[]
        for line in loco_logs_list[i]:
            m = re.match(loc_pat, line)
            if not m: continue
            ts,x,y,z,d = m.groups()
            evs.append((datetime.strptime(ts,"%Y-%m-%d %H:%M:%S.%f"),
                        np.array([float(x),float(y),float(z)]),
                        float(d)))
        loco_events.append(evs)

    TH=1.5
    for i in range(4):
        for j in range(i + 1, 4):
            if not loco_events[i] or not loco_events[j]:
                continue
            
            tot=0.0
            for ti,pi,dt in loco_events[i]:
                tj,pj,_ = min(loco_events[j], key=lambda e: abs((e[0]-ti).total_seconds()))
                if np.linalg.norm(pi-pj)<=TH:
                    tot+=dt
            if tot>0: G.add_edge(f"P{i+1}",f"P{j+1}", weight=tot)
    graphs.append(G); colors.append("green"); titles.append("Locomotion Sociogram")

    # Speaking (directed)
    G = nx.DiGraph(); G.add_nodes_from(pos)
    speak_events = [
      parse_intervals(speak_logs_list[i], r"(.+?): Speaking Event lasted ([\d\.]+) seconds")
      for i in range(4)
    ]
    for i in range(4):
        for j in range(4):
            if i==j: continue
            w = overlap(speak_events[i], speak_events[j])
            if w>0: G.add_edge(f"P{i+1}",f"P{j+1}", weight=w)
    graphs.append(G); colors.append("red"); titles.append("Speaking Sociogram")

    # 5) Plot & keep per-edge artists
    fig, axes = plt.subplots(1,3,figsize=(12,4))
    edge_artists = []   # list of lists per axis
    edge_weights = []

    for ax, G, c, title in zip(axes, graphs, colors, titles):
        ax.set_title(title); ax.axis("off")
        nx.draw_networkx_nodes(G, pos, ax=ax, node_size=600, node_color="lightblue")
        nx.draw_networkx_labels(G, pos, ax=ax)

        # normalize widths
        ws = [d["weight"] for _,_,d in G.edges(data=True)]
        mx = max(ws) if ws else 1

        arts, wts = [], []
        for u,v,d in G.edges(data=True):
            w = d["weight"]
            width = 1 + 4*(w/mx)
            if title=="Gaze Sociogram":
                line = Line2D([pos[u][0],pos[v][0]],
                              [pos[u][1],pos[v][1]],
                              linewidth=width, color=c, picker=5)
                ax.add_line(line)
                arts.append(line)
            else:
                arr = FancyArrowPatch(pos[u], pos[v],
                                     arrowstyle='-|>',
                                     connectionstyle='arc3,rad=0.15',
                                     mutation_scale=12,
                                     linewidth=width, color=c,
                                     picker=5)
                ax.add_patch(arr)
                arts.append(arr)
            wts.append(w)

        edge_artists.append(arts)
        edge_weights.append(wts)

    # 6) Hover tooltips
    def on_move(event):
        for ax, arts, wts in zip(axes, edge_artists, edge_weights):
            if event.inaxes is not ax:
                if hasattr(ax, 'tooltip'):
                    ax.tooltip.remove(); del ax.tooltip
                continue
            hit = False
            for art, w in zip(arts, wts):
                cont, _ = art.contains(event)
                if cont:
                    if hasattr(ax,'tooltip'):
                        ax.tooltip.remove()
                    ax.tooltip = ax.annotate(f"{w:.1f}s",
                                             xy=(event.xdata, event.ydata),
                                             xytext=(5,5),
                                             textcoords='offset points',
                                             bbox=dict(boxstyle='round,pad=0.2',
                                                       fc='yellow',alpha=0.7))
                    fig.canvas.draw_idle()
                    hit = True
                    break
            if not hit and hasattr(ax,'tooltip'):
                ax.tooltip.remove(); del ax.tooltip
                fig.canvas.draw_idle()

    win = tk.Toplevel(root)
    win.title("Sociograms")
    canvas = FigureCanvasTkAgg(fig, master=win)
    canvas.mpl_connect('motion_notify_event', on_move)
    canvas.draw()
    canvas.get_tk_widget().pack(fill="both", expand=True)

def show_actual_sociograms():
    MAX_ROWS_PER_CLUSTER = 10

    gaze_idxs  = [int(dd[0].get().split()[0]) - 1 for dd in participant_dropdowns]
    loc_idxs   = [int(dd[1].get().split()[0]) - 1 for dd in participant_dropdowns]
    speak_idxs = [int(dd[2].get().split()[0]) - 1 for dd in participant_dropdowns]

    def parse_raw_intervals(df, cluster_idx, pattern, timestamp_col="Start Times", dur_col="Durations"):
        df = df[df["Cluster_GMM"] == cluster_idx].head(MAX_ROWS_PER_CLUSTER)
        df[timestamp_col] = df[timestamp_col].apply(ast.literal_eval)
        df[dur_col] = df[dur_col].apply(ast.literal_eval)
        intervals = []
        for _, row in df.iterrows():
            for ts, dur in zip(row[timestamp_col], row[dur_col]):
                start = datetime.fromtimestamp(ts)
                end = start + timedelta(seconds=dur)
                intervals.append((start, end))
        return intervals

    def overlap(a, b):
        total = 0
        for s1, e1 in a:
            for s2, e2 in b:
                dt = (min(e1, e2) - max(s1, s2)).total_seconds()
                if dt > 0:
                    total += dt
        return total

    pos = {"P1": (0, 1), "P2": (1, 1), "P3": (0, 0), "P4": (1, 0)}
    graphs, colors, titles = [], [], []

    gaze_df = pd.read_csv("hover_gmm_user_clustering_results.csv")
    fft_df = pd.read_csv("hover_user_object_fft.csv")
    gaze_df["Cluster_GMM"] = pd.to_numeric(gaze_df["Cluster_GMM"], errors="coerce")
    fft_df["Group ID"] = fft_df["Group ID"].astype(str)

    G = nx.Graph(); G.add_nodes_from(pos)
    gaze_intervals = []
    for idx in gaze_idxs:
        group_ids = gaze_df[gaze_df["Cluster_GMM"] == idx].head(MAX_ROWS_PER_CLUSTER)["Group ID"].unique()
        durations = fft_df[fft_df["Group ID"].isin(group_ids)]["Mean Duration"].tolist()
        base_time = datetime.now()
        intervals = [(base_time + timedelta(seconds=i), base_time + timedelta(seconds=i+dur)) for i, dur in enumerate(durations)]
        gaze_intervals.append(intervals)

    for i in range(4):
        for j in range(i + 1, 4):
            w = overlap(gaze_intervals[i], gaze_intervals[j])
            if w > 0:
                G.add_edge(f"P{i+1}", f"P{j+1}", weight=w)
    graphs.append(G); colors.append("blue"); titles.append("Actual Gaze Sociogram")

    loc_df = pd.read_csv("loc_gmm_clustering_results.csv", low_memory=False)
    loc_df["Cluster_GMM"] = pd.to_numeric(loc_df["Cluster_GMM"], errors="coerce")
    loc_df["Timestamps"] = loc_df["Timestamps"].apply(ast.literal_eval)
    loc_df["X"] = loc_df["X"].apply(ast.literal_eval)
    loc_df["Y"] = loc_df["Y"].apply(ast.literal_eval)
    loc_df["Z"] = loc_df["Z"].apply(ast.literal_eval)

    G = nx.Graph(); G.add_nodes_from(pos)
    loco_events = []
    for idx in loc_idxs:
        events = []
        df = loc_df[loc_df["Cluster_GMM"] == idx].head(MAX_ROWS_PER_CLUSTER)
        for _, row in df.iterrows():
            ts, xs, ys, zs = row["Timestamps"], row["X"], row["Y"], row["Z"]
            for i in range(1, len(ts)):
                start = datetime.fromtimestamp(ts[i-1])
                dur = ts[i] - ts[i-1]
                position = np.array([xs[i], ys[i], zs[i]])
                events.append((start, position, dur))
        loco_events.append(events)

    TH = 1.5
    for i in range(4):
        for j in range(i + 1, 4):
            if not loco_events[i] or not loco_events[j]:
                continue
            tot = 0.0
            for ti, pi, dt in loco_events[i]:
                tj, pj, _ = min(loco_events[j], key=lambda e: abs((e[0] - ti).total_seconds()))
                if np.linalg.norm(pi - pj) <= TH:
                    tot += dt
            if tot > 0:
                G.add_edge(f"P{i+1}", f"P{j+1}", weight=tot)
    graphs.append(G); colors.append("green"); titles.append("Actual Locomotion Sociogram")

    speak_df = pd.read_csv("gmm_clustering_results_speaking.csv")
    speak_df["Cluster_GMM"] = pd.to_numeric(speak_df["Cluster_GMM"], errors="coerce")

    G = nx.DiGraph(); G.add_nodes_from(pos)
    speak_intervals = [
        parse_raw_intervals(speak_df, idx, None)
        for idx in speak_idxs
    ]

    for i in range(4):
        for j in range(4):
            if i == j:
                continue
            w = overlap(speak_intervals[i], speak_intervals[j])
            if w > 0:
                G.add_edge(f"P{i+1}", f"P{j+1}", weight=w)
    graphs.append(G); colors.append("red"); titles.append("Actual Speaking Sociogram")

    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    edge_artists, edge_weights = [], []

    for ax, G, c, title in zip(axes, graphs, colors, titles):
        ax.set_title(title); ax.axis("off")
        nx.draw_networkx_nodes(G, pos, ax=ax, node_size=600, node_color="lightblue")
        nx.draw_networkx_labels(G, pos, ax=ax)

        ws = [d["weight"] for _, _, d in G.edges(data=True)]
        mx = max(ws) if ws else 1
        arts, wts = [], []
        for u, v, d in G.edges(data=True):
            w = d["weight"]
            width = 1 + 4 * (w / mx)
            if title == "Actual Gaze Sociogram":
                line = Line2D([pos[u][0], pos[v][0]],
                              [pos[u][1], pos[v][1]],
                              linewidth=width, color=c, picker=5)
                ax.add_line(line)
                arts.append(line)
            else:
                arr = FancyArrowPatch(pos[u], pos[v],
                                      arrowstyle='-|>',
                                      connectionstyle='arc3,rad=0.15',
                                      mutation_scale=12,
                                      linewidth=width, color=c,
                                      picker=5)
                ax.add_patch(arr)
                arts.append(arr)
            wts.append(w)
        edge_artists.append(arts)
        edge_weights.append(wts)

    def on_move(event):
        for ax, arts, wts in zip(axes, edge_artists, edge_weights):
            if event.inaxes is not ax:
                if hasattr(ax, 'tooltip'):
                    ax.tooltip.remove(); del ax.tooltip
                continue
            hit = False
            for art, w in zip(arts, wts):
                cont, _ = art.contains(event)
                if cont:
                    if hasattr(ax, 'tooltip'):
                        ax.tooltip.remove()
                    ax.tooltip = ax.annotate(f"{w:.1f}s",
                                             xy=(event.xdata, event.ydata),
                                             xytext=(5, 5),
                                             textcoords='offset points',
                                             bbox=dict(boxstyle='round,pad=0.2',
                                                       fc='yellow', alpha=0.7))
                    fig.canvas.draw_idle()
                    hit = True
                    break
            if not hit and hasattr(ax, 'tooltip'):
                ax.tooltip.remove(); del ax.tooltip
                fig.canvas.draw_idle()

    win = tk.Toplevel(root)
    win.title("Actual Sociograms")
    canvas = FigureCanvasTkAgg(fig, master=win)
    canvas.mpl_connect('motion_notify_event', on_move)
    canvas.draw()
    canvas.get_tk_widget().pack(fill="both", expand=True)




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
    

def generate_speaking_log(cluster_idx):
    try:
        df = pd.read_csv("gmm_clustering_results_speaking.csv")
        df["Start Times"] = df["Start Times"].apply(ast.literal_eval)
        df["Durations"] = df["Durations"].apply(ast.literal_eval)
        cluster_df = df[df["Cluster_GMM"] == cluster_idx]

        all_starts, all_durs = [], []
        for _, row in cluster_df.iterrows():
            all_starts += row["Start Times"]
            all_durs += row["Durations"]

        if not all_starts or not all_durs:
            return [f"No valid speaking data for cluster {cluster_idx}."]

        start_hist, edges = np.histogram(all_starts, bins='auto', density=True)
        start_smoothed = np.fft.irfft(np.fft.rfft(start_hist)[:4], n=len(start_hist))
        start_probs = np.maximum(start_smoothed, 0)
        start_probs /= start_probs.sum()
        start_bins = (edges[:-1] + edges[1:]) / 2
        sampled_starts = np.random.choice(start_bins, size=len(all_starts), p=start_probs)

        durs = np.array(all_durs)
        durs = durs[(durs > 0.01) & (durs < 10)]
        if len(durs) < 10:
            return [f"Not enough valid durations in cluster {cluster_idx}."]

        xs = np.linspace(0.01, 10.0, 500)
        kde = gaussian_kde(durs)
        probs = kde(xs)
        probs = np.maximum(probs, 0)
        probs /= probs.sum()
        sampled_durs = np.random.choice(xs, size=len(durs), p=probs)

        base_time = datetime.now()
        logs = []
        for ts, dur in zip(sampled_starts, sampled_durs):
            time_str = (base_time + timedelta(seconds=ts)).strftime("%Y-%m-%d %H:%M:%S")
            logs.append(f"{time_str}: Speaking Event lasted {dur:.2f} seconds")
        return logs

    except Exception as e:
        return [f"Error generating speaking log: {e}"]

    

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



    

    
# --- Tooltip Helper Class ---
class ToolTip:
    def __init__(self, widget, text):
        self.widget = widget
        self.text = text
        self.tooltip_window = None
        self.widget.bind("<Enter>", self.show_tooltip)
        self.widget.bind("<Leave>", self.hide_tooltip)

    def show_tooltip(self, event=None):
        if self.tooltip_window:
            return
        x = self.widget.winfo_rootx() + 20
        y = self.widget.winfo_rooty() + 20
        self.tooltip_window = tw = tk.Toplevel(self.widget)
        tw.wm_overrideredirect(True)
        tw.wm_geometry(f"+{x}+{y}")
        label = tk.Label(tw, text=self.text, justify='left', background="#ffffe0", relief='solid', borderwidth=1, font=("Helvetica", 10))
        label.pack(ipadx=1)

    def hide_tooltip(self, event=None):
        if self.tooltip_window:
            self.tooltip_window.destroy()
            self.tooltip_window = None

# --- UI: Create participant panel ---
def create_participant_panel(parent, participant_num):
    frame = ttk.LabelFrame(parent, text=f"Participant {participant_num}", padding=(10, 5))
    frame.pack(padx=5, pady=10, fill="x")

    gaze_label = ttk.Label(frame, text="Gaze Activity")
    gaze_label.pack(anchor="w")
    gaze_dd = ttk.Combobox(frame, values=["1 (Low)", "2 (High)", "3 (Moderate)"], state="readonly")
    gaze_dd.current(0)
    gaze_dd.pack(anchor="w", fill="x")
    ToolTip(gaze_label, "Instances count, mean/std of timestamps and duration,\nvirtual object looked at, first three non-DC Fourier frequency\ncomponents of timestamps")

    loc_label = ttk.Label(frame, text="Locomotion Activity")
    loc_label.pack(anchor="w", pady=(10, 0))
    loc_dd = ttk.Combobox(frame, values=["1 (Consistent)", "2 (Variable)", "3 (Dynamic)", "4 (Stable)"], state="readonly")
    loc_dd.current(0)
    loc_dd.pack(anchor="w", fill="x")
    ToolTip(loc_label, "Instances count, mean/std of timestamps and X, Y, Z coordinates,\nrange of X, Y, Z, total distance/time, mean/max speed and acceleration,\nmean jerk, path tortuosity, idle fraction,\nfirst three non-DC Fourier frequency components.")

    speak_label = ttk.Label(frame, text="Speaking Activity")
    speak_label.pack(anchor="w", pady=(10, 0))
    speak_dd = ttk.Combobox(frame, values=["1 (Frequent)", "2 (Infrequent)", "3 (Moderate)"], state="readonly")
    speak_dd.current(0)
    speak_dd.pack(anchor="w", fill="x")
    ToolTip(speak_label, "Instances count, mean/std of timestamps and duration,\nfirst three non-DC Fourier frequency components of timestamps")

    return (gaze_dd, loc_dd, speak_dd)



def merge_simulated_clusters(participant_dropdowns, last_sim_logs):
    """
    Merge user-input cluster config with simulated logs into a DataFrame.
    Each participant gets their chosen cluster index and the actual logs generated.
    """
    data = []
    for i, (dropdowns, logs) in enumerate(zip(participant_dropdowns, last_sim_logs), start=1):
        gaze_idx = int(dropdowns[0].get().split()[0]) - 1
        loc_idx = int(dropdowns[1].get().split()[0]) - 1
        speak_idx = int(dropdowns[2].get().split()[0]) - 1

        data.append({
            "Group ID": i,
            "Participant ID": i,
            "Gaze Cluster": gaze_idx,
            "Loc Cluster": loc_idx,
            "Speaking Cluster": speak_idx,
            "Gaze Log": logs["gaze"],
            "Loc Log": logs["loc"],
            "Speaking Log": logs["speak"]
        })

    df = pd.DataFrame(data)
    df.to_csv("merged_simulated_cluster_config.csv", index=False)
    return df

def merge_raw_clusters_from_ui(participant_dropdowns):
    """
    Merge raw cluster information based on user-selected cluster IDs.
    Pulls matching entries from the raw CSV files for all modalities.
    """
    gaze_idxs = [int(dd[0].get().split()[0]) - 1 for dd in participant_dropdowns]
    loc_idxs = [int(dd[1].get().split()[0]) - 1 for dd in participant_dropdowns]
    speak_idxs = [int(dd[2].get().split()[0]) - 1 for dd in participant_dropdowns]

    # Load raw datasets
    gaze_df = pd.read_csv("hover_gmm_user_clustering_results.csv")
    loc_df = pd.read_csv("loc_gmm_clustering_results.csv", low_memory=False)
    speak_df = pd.read_csv("gmm_clustering_results_speaking.csv")

    gaze_df["Cluster_GMM"] = pd.to_numeric(gaze_df["Cluster_GMM"], errors="coerce")
    loc_df["Cluster_GMM"] = pd.to_numeric(loc_df["Cluster_GMM"], errors="coerce")
    speak_df["Cluster_GMM"] = pd.to_numeric(speak_df["Cluster_GMM"], errors="coerce")

    rows = []
    for i in range(4):
        gaze_rows = gaze_df[gaze_df["Cluster_GMM"] == gaze_idxs[i]]
        loc_rows = loc_df[loc_df["Cluster_GMM"] == loc_idxs[i]]
        speak_rows = speak_df[speak_df["Cluster_GMM"] == speak_idxs[i]]

        rows.append({
            "Group ID": i+1,
            "Participant ID": i+1,
            "Gaze Cluster": gaze_idxs[i],
            "Loc Cluster": loc_idxs[i],
            "Speaking Cluster": speak_idxs[i],
            "Gaze Raw Matches": len(gaze_rows),
            "Loc Raw Matches": len(loc_rows),
            "Speaking Raw Matches": len(speak_rows)
        })

    df = pd.DataFrame(rows)
    df.to_csv("merged_cluster_config.csv", index=False)
    return df

# --- Start simulation logic ---
def start_simulation():
    global last_sim_logs
    last_sim_logs = []             # clear cache on each new sim run
    status_label.config(text="Status: Running")
    log_text.delete(1.0, tk.END)

    log_text.insert(tk.END, f"Simulation started for 4 participants:\n")
    log_text.insert(tk.END, "-"*30 + "\n")

    for idx, (gaze_dd, loc_dd, speak_dd) in enumerate(participant_dropdowns, start=1):
        gaze_cluster = int(gaze_dd.get().split()[0]) - 1
        loc_cluster = int(loc_dd.get().split()[0]) - 1
        speak_cluster = int(speak_dd.get().split()[0]) - 1

        log_text.insert(tk.END, f"\n[Participant {idx}]\n")
        log_text.insert(tk.END, f"  Gaze Cluster: {gaze_cluster + 1}\n")
        log_text.insert(tk.END, f"  Loc Cluster: {loc_cluster + 1}\n")
        log_text.insert(tk.END, f"  Speaking Cluster: {speak_cluster + 1}\n")

        gaze_logs = generate_gaze_log(gaze_cluster)
        loc_logs = generate_location_log(loc_cluster)
        speak_logs = generate_speaking_log(speak_cluster)
          # cache them for the sociogram
        last_sim_logs.append({
            "gaze":  gaze_logs,
            "loc":   loc_logs,
            "speak": speak_logs
        })

        log_text.insert(tk.END, "\n[Gaze Log]\n" + "\n".join(gaze_logs) + "\n")
        log_text.insert(tk.END, "\n[Locomotion Log]\n" + "\n".join(loc_logs) + "\n")
        log_text.insert(tk.END, "\n[Speaking Log]\n" + "\n".join(speak_logs) + "\n")

    log_text.see(tk.END)
 
    

def stop_simulation():
    status_label.config(text="Status: Stopped")
    log_text.insert(tk.END, "Simulation stopped.\n")
    log_text.insert(tk.END, "-"*30 + "\n")
    log_text.see(tk.END)

# --- Build the UI ---
root = tk.Tk()
root.title("Simulator UI - 4 Participants")
root.geometry("1100x700")

main_frame = ttk.Frame(root)
main_frame.pack(fill="both", expand=True, padx=10, pady=10)

# Make control panel scrollable
control_canvas = tk.Canvas(main_frame)
scrollbar = ttk.Scrollbar(main_frame, orient="vertical", command=control_canvas.yview)
scrollable_control = ttk.Frame(control_canvas)

scrollable_control.bind(
    "<Configure>", lambda e: control_canvas.configure(scrollregion=control_canvas.bbox("all"))
)

control_canvas.create_window((0, 0), window=scrollable_control, anchor="nw")
control_canvas.configure(yscrollcommand=scrollbar.set, width=400)

control_canvas.pack(side="left", fill="y", expand=False)
scrollbar.pack(side="left", fill="y")

label = ttk.Label(scrollable_control, text="Simulation Control Panel", font=("Helvetica", 16))
label.pack(pady=10)

participant_dropdowns = []
for i in range(4):
    dropdowns = create_participant_panel(scrollable_control, i + 1)
    participant_dropdowns.append(dropdowns)

start_button = ttk.Button(scrollable_control, text="Start Simulation", command=start_simulation)
start_button.pack(pady=5)

stop_button = ttk.Button(scrollable_control, text="Stop Simulation", command=stop_simulation)
stop_button.pack(pady=5)

status_label = ttk.Label(scrollable_control, text="Status: Idle", font=("Helvetica", 14))
status_label.pack(pady=10)

raw_button = ttk.Button(scrollable_control,text="Show Raw Histograms",command=show_histograms_raw)
raw_button.pack(pady=5)

log_button = ttk.Button(scrollable_control,text="Show Log-based Histograms",command=show_histograms_user)
log_button.pack(pady=5)

socio_button = ttk.Button(scrollable_control,text="Show Sociograms",command=show_sociograms)
socio_button.pack(pady=5)

actual_button = ttk.Button(scrollable_control, text="Show Actual Sociograms", command=show_actual_sociograms)
actual_button.pack(pady=5)


log_frame = ttk.Frame(main_frame)
log_frame.pack(side="right", fill="both", expand=True, padx=10, pady=10)

log_label = ttk.Label(log_frame, text="Log Output", font=("Helvetica", 14))
log_label.pack(anchor="w")

log_text = tk.Text(log_frame, height=30, width=70)
log_text.pack(fill="both", expand=True)

root.protocol("WM_DELETE_WINDOW", on_closing)

root.mainloop()
