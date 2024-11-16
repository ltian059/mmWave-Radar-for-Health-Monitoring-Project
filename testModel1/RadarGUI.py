import tkinter as tk
from tkinter import font
import time

import Configuration
import utils

from utils import update, close_ports
from Configuration import serialConfig,parseConfigFile
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

configFileName = Configuration.configFileName
class RadarGUI:
    def __init__(self, root, model, device):
        self.CLIport = None
        self.Dataport = None
        self.is_recording = False
        self.root = root
        self.model = model
        self.device = device
        self.configFileName = configFileName
        # Set up the serial connection and radar configuration parameters
        configParameters = parseConfigFile(configFileName)
        self.configParameters = configParameters

        self.init_root()

        self.init_plt()

        self.init_gadgets()

    def init_root(self):
        self.root.title("Radar GUI")
        self.root.geometry("800x600")  # Initial window size
        self.root.minsize(400, 300)  # Set minimum window size
        # Configure row and column weights for resizing
        self.root.grid_rowconfigure(0, weight=1)
        self.root.grid_rowconfigure(1, weight=0)  # Button row
        self.root.grid_columnconfigure(0, weight=1)
    def init_plt(self):
        # Create a Matplotlib figure for 3D plotting
        self.fig = plt.Figure(figsize=(8, 8))
        self.ax = self.fig.add_subplot(111, projection='3d')
        self.ax.set_xticks([])  # Hide ticks
        self.ax.set_yticks([])
        self.ax.set_zticks([])

        # Adjust subplot parameters to reduce padding and increase plot area
        self.fig.subplots_adjust(left=0.05, right=0.95, top=0.95, bottom=0.05)

        # Embed Matplotlib figure in tkinter and make it fill the upper area
        self.canvas = FigureCanvasTkAgg(self.fig, master=self.root)
        self.canvas.get_tk_widget().grid(row=0, column=0, sticky="nsew")  # Enable resizing with sticky.
    def init_gadgets(self):
        self.default_font = font.Font(family="Arial", size=16, weight="normal")
        self.indicator_font = font.Font(family="Arial", size=18, weight="normal")

        # Fonts for buttons and labels
        self.default_font = font.Font(family="Arial", size=16, weight="normal")
        self.indicator_font = font.Font(family="Arial", size=18, weight="normal")

        # Start Detecting Button
        self.start_button = tk.Button(self.root, text="Start Detecting", command=self.toggle_recording,
                                      font=self.default_font, bg="white", fg="black",
                                      activebackground="#f0f0f0", activeforeground="black", borderwidth=2)
        self.start_button.grid(row=1, column=0, pady=10, sticky="ew")  # Button scales horizontally with the window

        # Fall Indicator Label
        self.fall_indicator = tk.Label(self.root, text="Monitoring...", bg="green", fg="white", font=self.indicator_font,
                                       width=20, height=2, borderwidth=2, relief="solid")
        self.fall_indicator.grid(row=2, column=0, pady=10, sticky="ew")  # Label scales horizontally with the window

    def update_scatter_plot_with_colors(self, points_with_ids, size=2):
        """Update the scatter plot with different colors based on target IDs."""
        points = points_with_ids[:, :3]  # Extract X, Y, Z coordinates
        target_ids = points_with_ids[:, 4].astype(int)  # Extract the target IDs

        # Define color mapping based on target ID, including radar point
        color_map = {
            253: 'red',  # Red for SNR too weak
            254: 'blue',  # Blue for points outside boundary
            255: 'gray',  # Gray for noise points
            888: 'white',  # White for radar location
        }# Default color for valid points
        colors = np.array([
            'green' if target_id <= 252 else color_map.get(target_id, 'white')
            for target_id in target_ids
        ])

        # Clear previous plot and re-plot
        self.ax.cla()
        self.ax.set_xticks([])  # Hide ticks
        self.ax.set_yticks([])
        self.ax.set_zticks([])
        # Plot points with color mapping
        self.ax.scatter(points[:, 0], points[:, 1], points[:, 2], c=colors, s=size)
        # Refresh the canvas
        self.canvas.draw()

    def on_start(self):
        Configuration.csv_file_path_timestamp = utils.generate_csv_title_timestamp()
        self.CLIport, self.Dataport = serialConfig(configFileName)
        config = [line.rstrip('\r\n') for line in open(self.configFileName)]
        for i in config:
            self.CLIport.write((i + '\n').encode())
            print(i)
            time.sleep(0.03)
        print("Fall Detection started.")
        self.start_button.config(text="Stop Detecting", state="active",fg="white", bg="#d9534f", activebackground="#c9302c")
        self.call_update()

    def on_stop(self):
        print("Fall Detection stopped.")
        self.reset_indicator()
        # close ports
        close_ports(self.CLIport, self.Dataport)
        self.start_button.config(text="Start Detecting", state="active", fg="black", bg="white", activebackground="#f0f0f0")

    def call_update(self):
        if self.is_recording:
            update(self.Dataport, self.configParameters, self.model, self.device, self)
            self.root.after(3, self.call_update) # call update every 10 ms

    def toggle_recording(self):
        self.is_recording = not self.is_recording
        if self.is_recording:
            self.start_button.config(text="Starting...", state="disabled", fg="white", bg="#d9534f", activebackground="#c9302c")
            self.root.after(50, self.on_start)
        else:
            self.start_button.config(text="Stopping...", state="disabled", fg="white", bg="#d9534f", activebackground="#c9302c")
            self.root.after(50, self.on_stop)


    def update_indicator(self, detected_activity):
        """
        Updates the indicator based on the detected activity.
        """
        if detected_activity == "Falling":
            self.fall_indicator.config(text="FALL DETECTED", bg="red", fg="white")
            # messagebox.showwarning("Alert", "FALL DETECTED")
        elif detected_activity == "Sitting":
            self.fall_indicator.config(text="Sitting", bg="yellow", fg="black")
        elif detected_activity == "Standing / \nWalking":
            self.fall_indicator.config(text="Standing/Walking", bg="yellow", fg="black")
        else:
            self.fall_indicator.config(text="Monitoring...", bg="green", fg="white")

    def reset_indicator(self):
        """
        Resets the indicator to its original state.
        """
        self.fall_indicator.config(text="Monitoring...", bg="green", fg="white")
