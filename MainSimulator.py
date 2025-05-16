import tkinter as tk
from tkinter import ttk
import pandas as pd
import numpy as np
import ast
from datetime import datetime, timedelta


def generate_location_log(cluster_idx):
    """
    Simulates location movement logs based on cluster index from loc_gmm_clustering_results.csv.
    Returns a list of log strings.
    """
    try:
        df = pd.read_csv("loc_gmm_clustering_results.csv", low_memory=False)
        df["Timestamps"] = df["Timestamps"].apply(ast.literal_eval)
        df["X"] = df["X"].apply(ast.literal_eval)
        df["Y"] = df["Y"].apply(ast.literal_eval)
        df["Z"] = df["Z"].apply(ast.literal_eval)
        cluster_df = df[df["Cluster_GMM"] == cluster_idx]
        diff_list = []

        for _, row in cluster_df.iterrows():
            ts, xs, ys, zs = row["Timestamps"], row["X"], row["Y"], row["Z"]
            for i in range(1, len(ts)):
                diff_list.append([ts[i]-ts[i-1], xs[i]-xs[i-1], ys[i]-ys[i-1], zs[i]-zs[i-1]])
        
        if not diff_list:
            return ["No movement data available for this cluster."]

        diff_array = np.array(diff_list)
        mean_diff = np.mean(diff_array, axis=0)
        cov_diff = np.cov(diff_array.T)
        synthetic_diffs = np.random.multivariate_normal(mean_diff, cov_diff, size=10)
        logs, ts, x, y, z = [], [0], [0], [0], [0]

        for dt, dx, dy, dz in synthetic_diffs:
            ts.append(ts[-1]+dt)
            x.append(x[-1]+dx)
            y.append(y[-1]+dy)
            z.append(z[-1]+dz)

        base_time = datetime.now()
        for t, xi, yi, zi in zip(ts, x, y, z):
            time_str = (base_time + timedelta(seconds=t)).strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]
            logs.append(f"{time_str}: Position - X: {xi:.2f}, Y: {yi:.2f}, Z: {zi:.2f}")
        return logs
    except Exception as e:
        return [f"Error generating location log: {e}"]
    

def generate_speaking_log(cluster_idx):
    """
    Simulates speaking logs based on gmm_clustering_results_speaking.csv.
    Returns a list of log strings.
    """
    try:
        df = pd.read_csv("gmm_clustering_results_speaking.csv")
        df.columns = df.columns.str.strip()  # Strip spaces just in case
        df["Start Times"] = df["Start Times"].apply(ast.literal_eval)
        cluster_df = df[df["Cluster_GMM"] == cluster_idx]
        if cluster_df.empty:
            return ["No speaking data available for this cluster."]
        
        log_lines = []
        for _, row in cluster_df.iterrows():
            for ts in row["Start Times"]:
                time_str = (datetime.now() + timedelta(seconds=ts)).strftime("%Y-%m-%d %H:%M:%S")
                log_lines.append(f"{time_str}: Speaking Event")
        return log_lines or ["No timestamps found in speaking cluster."]
    except Exception as e:
        return [f"Error generating speaking log: {e}"]
    

def generate_gaze_log(cluster_idx):
    try:
        # Load cluster mapping
        cluster_df = pd.read_csv("hover_gmm_user_clustering_results.csv")
        cluster_df.columns = cluster_df.columns.str.strip()
        cluster_df["Group ID"] = cluster_df["Group ID"].astype(str)
        cluster_df["Cluster_GMM"] = pd.to_numeric(cluster_df["Cluster_GMM"], errors="coerce")

        # Filter Group IDs that match the selected cluster
        group_ids = cluster_df[cluster_df["Cluster_GMM"] == cluster_idx]["Group ID"].unique()

        if len(group_ids) == 0:
            return [f"No gaze groups found for cluster {cluster_idx}."]

        # Load gaze features
        fft_df = pd.read_csv("hover_user_object_fft.csv")
        fft_df.columns = fft_df.columns.str.strip()
        fft_df["Group ID"] = fft_df["Group ID"].astype(str)

        # Filter rows from fft_df matching selected group_ids
        filtered = fft_df[fft_df["Group ID"].isin(group_ids)]
        if filtered.empty:
            return [f"No gaze data available for cluster {cluster_idx}."]

        log_lines = []
        for _, row in filtered.iterrows():
            obj = row["Virtual Object"]
            duration = row["Mean Duration"]
            start_offset = row["Mean Start Time"]
            fft = [row["FFT1"], row["FFT2"], row["FFT3"]]
            time_str = (datetime.now() + timedelta(seconds=start_offset)).strftime("%Y-%m-%d %H:%M:%S")
            log_lines.append(f"{time_str}: Gazed at {obj} for {duration:.2f}s | FFTs: {fft}")

        return log_lines

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

# --- Start simulation logic ---
def start_simulation():
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

log_frame = ttk.Frame(main_frame)
log_frame.pack(side="right", fill="both", expand=True, padx=10, pady=10)

log_label = ttk.Label(log_frame, text="Log Output", font=("Helvetica", 14))
log_label.pack(anchor="w")

log_text = tk.Text(log_frame, height=30, width=70)
log_text.pack(fill="both", expand=True)

root.mainloop()
