import tkinter as tk
from tkinter import ttk
import pandas as pd
import numpy as np
import ast
from datetime import datetime, timedelta

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

# --- Main UI Code ---
def start_simulation():
    status_label.config(text="Status: Running")
    log_text.insert(tk.END, f"Simulation started with:\n")
    log_text.insert(tk.END, f"  Gaze Activity: {gaze_dropdown.get()}\n")
    log_text.insert(tk.END, f"  Locomotion Activity: {loc_dropdown.get()}\n")
    log_text.insert(tk.END, f"  Speaking Activity: {speaking_dropdown.get()}\n")
    log_text.insert(tk.END, "-"*30 + "\n")
    log_text.see(tk.END)

def stop_simulation():
    status_label.config(text="Status: Stopped")
    log_text.insert(tk.END, "Simulation stopped.\n")
    log_text.insert(tk.END, "-"*30 + "\n")
    log_text.see(tk.END)

root = tk.Tk()
root.title("Simulator UI")
root.geometry("1000x600")

main_frame = ttk.Frame(root)
main_frame.pack(fill="both", expand=True, padx=10, pady=10)

control_frame = ttk.Frame(main_frame)
control_frame.pack(side="left", fill="y", padx=10, pady=10)

label = ttk.Label(control_frame, text="Simulation Control Panel", font=("Helvetica", 16))
label.pack(pady=10)

selection_frame = ttk.Frame(control_frame)
selection_frame.pack(pady=10)

# --- Gaze Activity ---
gaze_label = ttk.Label(selection_frame, text="Gaze Activity")
gaze_label.pack(anchor="w")
ToolTip(gaze_label, "Instances count, mean/std of timestamps and duration,\nvirtual object looked at, first three non-DC Fourier frequency\ncomponents of timestamps")

gaze_options = ["1 (Low)", "2 (High)", "3 (Moderate)"]
gaze_dropdown = ttk.Combobox(selection_frame, values=gaze_options, state="readonly")
gaze_dropdown.current(0)
gaze_dropdown.pack(anchor="w", fill="x")

# --- Locomotion Activity ---
loc_label = ttk.Label(selection_frame, text="Locomotion Activity")
loc_label.pack(anchor="w", pady=(10, 0))
ToolTip(loc_label, "Instances count, mean/std of timestamps and X, Y, Z coordinates,\nrange of X, Y, Z, total distance/time, mean/max speed and acceleration,\nmean jerk, path tortuosity, idle fraction,\nfirst three non-DC Fourier frequency components.")

loc_options = ["1 (Consistent)", "2 (Variable)", "3 (Dynamic)", "4 (Stable)"]
loc_dropdown = ttk.Combobox(selection_frame, values=loc_options, state="readonly")
loc_dropdown.current(0)
loc_dropdown.pack(anchor="w", fill="x")

# --- Speaking Activity ---
speaking_label = ttk.Label(selection_frame, text="Speaking Activity")
speaking_label.pack(anchor="w", pady=(10, 0))
ToolTip(speaking_label, "Instances count, mean/std of timestamps and duration,\nfirst three non-DC Fourier frequency components of timestamps")

speaking_options = ["1 (Frequent)", "2 (Infrequent)", "3 (Moderate)"]
speaking_dropdown = ttk.Combobox(selection_frame, values=speaking_options, state="readonly")
speaking_dropdown.current(0)
speaking_dropdown.pack(anchor="w", fill="x")

# Buttons
start_button = ttk.Button(control_frame, text="Start Simulation", command=start_simulation)
start_button.pack(pady=5)

stop_button = ttk.Button(control_frame, text="Stop Simulation", command=stop_simulation)
stop_button.pack(pady=5)

status_label = ttk.Label(control_frame, text="Status: Idle", font=("Helvetica", 14))
status_label.pack(pady=10)

# Right Log Panel
log_frame = ttk.Frame(main_frame)
log_frame.pack(side="right", fill="both", expand=True, padx=10, pady=10)

log_label = ttk.Label(log_frame, text="Log Output", font=("Helvetica", 14))
log_label.pack(anchor="w")

log_text = tk.Text(log_frame, height=30, width=60)
log_text.pack(fill="both", expand=True)

root.mainloop()
