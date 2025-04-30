import tkinter as tk
from tkinter import ttk
import pandas as pd
import numpy as np
import ast
from datetime import datetime, timedelta

def start_simulation():
    status_label.config(text="Status: Running")
    log_text.insert(tk.END, f"Simulation started with:\n")
    log_text.insert(tk.END, f"  Gaze Activity: {gaze_var.get()}\n")
    log_text.insert(tk.END, f"  Locomotion Activity: {loc_var.get()}\n")
    log_text.insert(tk.END, f"  Speaking Activity: {speaking_var.get()}\n")
    log_text.insert(tk.END, "-"*30 + "\n")
    log_text.see(tk.END)

def stop_simulation():
    status_label.config(text="Status: Stopped")
    log_text.insert(tk.END, "Simulation stopped.\n")
    log_text.insert(tk.END, "-"*30 + "\n")
    log_text.see(tk.END)

root = tk.Tk()
root.title("Simulator UI")
root.geometry("1000x600")  # Wider to fit more content

# Main Frame
main_frame = ttk.Frame(root)
main_frame.pack(fill="both", expand=True, padx=10, pady=10)

# Left Frame: Controls
control_frame = ttk.Frame(main_frame)
control_frame.pack(side="left", fill="y", padx=10, pady=10)

label = ttk.Label(control_frame, text="Simulation Control Panel", font=("Helvetica", 16))
label.pack(pady=10)

# Selection Frame
selection_frame = ttk.Frame(control_frame)
selection_frame.pack(pady=10)

# --- Gaze Model ---
gaze_label = ttk.Label(selection_frame, text="Gaze Activity")
gaze_label.pack(anchor="w")

gaze_var = tk.IntVar(value=1)
gaze_options = {
    1: "1 (Low)",
    2: "2 (High)",
    3: "3 (Moderate)"
}
for val, text in gaze_options.items():
    ttk.Radiobutton(selection_frame, text=text, variable=gaze_var, value=val).pack(anchor="w")

# --- Loc ---
loc_label = ttk.Label(selection_frame, text="Locomotion Activity")
loc_label.pack(anchor="w", pady=(10, 0))

loc_var = tk.IntVar(value=1)
loc_options = {
    1: "1 (Consistent)",
    2: "2 (Variable)",
    3: "3 (Dynamic)",
    4: "4 (Stable)"
}
for val, text in loc_options.items():
    ttk.Radiobutton(selection_frame, text=text, variable=loc_var, value=val).pack(anchor="w")

# --- Speaking ---
speaking_label = ttk.Label(selection_frame, text="Speaking Activity")
speaking_label.pack(anchor="w", pady=(10, 0))

speaking_var = tk.IntVar(value=1)
speaking_options = {
    1: "1 (Frequent)",
    2: "2 (Infrequent)",
    3: "3 (Moderate)"
}
for val, text in speaking_options.items():
    ttk.Radiobutton(selection_frame, text=text, variable=speaking_var, value=val).pack(anchor="w")

# --- Buttons ---
start_button = ttk.Button(control_frame, text="Start Simulation", command=start_simulation)
start_button.pack(pady=5)

stop_button = ttk.Button(control_frame, text="Stop Simulation", command=stop_simulation)
stop_button.pack(pady=5)

# --- Status ---
status_label = ttk.Label(control_frame, text="Status: Idle", font=("Helvetica", 14))
status_label.pack(pady=10)

# --- Right Frame: Log Screen ---
log_frame = ttk.Frame(main_frame)
log_frame.pack(side="right", fill="both", expand=True, padx=10, pady=10)

log_label = ttk.Label(log_frame, text="Log Output", font=("Helvetica", 14))
log_label.pack(anchor="w")

log_text = tk.Text(log_frame, height=30, width=60)
log_text.pack(fill="both", expand=True)

root.mainloop()
