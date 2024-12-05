import numpy as np
import pandas as pd
import torch
from PyQt5 import QtWidgets, QtCore, QtGui
import pyqtgraph.opengl as gl
from utils import readAndParseData14xx, fill_frame, save_to_csv, num_to_class, append_to_csv
import Configuration
# ------------------- GUI SET UP --------------------------------------------
class RadarGUI(QtWidgets.QMainWindow):
    def __init__(self, parent=None):
        super(RadarGUI, self).__init__(parent)
        self.is_recording = False

        # Set up central widget and layout
        central_widget = QtWidgets.QWidget()
        self.setCentralWidget(central_widget)
        layout = QtWidgets.QVBoxLayout(central_widget)

        # Set up the 3D scatter plot widget and add it to the layout
        self.scatter_widget = gl.GLViewWidget()
        layout.addWidget(self.scatter_widget)

        # Initialize the timer for resetting the fall indicator
        self.reset_timer = QtCore.QTimer(self)
        self.reset_timer.setSingleShot(True)
        self.reset_timer.timeout.connect(self.reset_fall_indicator)

        # Create and add a scatter plot item and a cube frame to the widget
        self.scatter = gl.GLScatterPlotItem()
        self.scatter_widget.addItem(self.scatter)
        cube_lines = self.create_cube(width=5, height=5, depth=3, y_translation=2.5)
        for line_item in cube_lines:
            self.scatter_widget.addItem(line_item)

        # Configure the camera for an isometric view
        self.scatter_widget.setCameraPosition(distance=15, elevation=30, azimuth=45)
        self.scatter_widget.opts['center'] = QtGui.QVector3D(-2, -0, -2)  # Adjust the center if needed
        self.scatter_widget.update()

        # Create occupancy grid
        self.create_occupancy_grid(cube_width=5, cube_height=3, cube_depth=5, grid_width=10, grid_height=10,
                                   spacing=0.5, cube_y_translation=0)

        # Bottom layout for button and fall indicator
        bottom_layout = QtWidgets.QHBoxLayout()
        bottom_layout.addStretch()  # Add a spacer on the left side

        # Create the Start Recording button
        self.start_recording_button = QtWidgets.QPushButton("Start Detecting")
        button_size = 250  # Square button size
        self.start_recording_button.setFixedSize(button_size, button_size)
        self.start_recording_button.setStyleSheet("QPushButton { font-size: 18pt; }")
        bottom_layout.addWidget(self.start_recording_button)

        # Modify the fall detection indicator (label)
        self.fall_indicator = QtWidgets.QLabel("Monitoring...")
        self.fall_indicator.setAlignment(QtCore.Qt.AlignCenter)
        self.fall_indicator.setFixedSize(button_size, button_size)
        self.fall_indicator.setStyleSheet(
            "QLabel { background-color: green; border: 1px solid black; font-size: 18pt; }")
        bottom_layout.addWidget(self.fall_indicator)

        bottom_layout.addStretch()  # Add a spacer on the right side

        # Add the bottom layout to the main vertical layout
        layout.addLayout(bottom_layout)

        # Connect the button click to the start_recording method
        self.start_recording_button.clicked.connect(self.start_recording)

    def start_recording(self):
        # Toggle the is_recording flag
        self.is_recording = not self.is_recording

        # Update button text based on the recording state
        if self.is_recording:
            self.start_recording_button.setText("Stop Detecting")
            print("Fall Detection started.")
        else:
            self.start_recording_button.setText("Start Detecting")
            print("Fall Detection stopped.")

    def create_occupancy_grid(self, cube_width, cube_height, cube_depth, grid_width, grid_height, spacing,
                              cube_y_translation):
        # Calculate the center of the cube in the x and y dimensions
        cube_center_x = 0
        cube_center_y = 2.5
        z_position = cube_y_translation - (cube_height / 2)
        grid_color = (0.5, 0.5, 0.5, 1)  # Light grey color for the grid lines
        lines = []
        grid_start_x = cube_center_x - (grid_width / 2)
        grid_start_y = cube_center_y - (grid_height / 2)

        # Horizontal and vertical lines
        for y in np.arange(grid_start_y, grid_start_y + grid_height + spacing, spacing):
            lines.append([np.array([grid_start_x, y, z_position], dtype=np.float32),
                          np.array([grid_start_x + grid_width, y, z_position], dtype=np.float32)])
        for x in np.arange(grid_start_x, grid_start_x + grid_width + spacing, spacing):
            lines.append([np.array([x, grid_start_y, z_position], dtype=np.float32),
                          np.array([x, grid_start_y + grid_height, z_position], dtype=np.float32)])

        # Create line plot items for each line in the grid
        for line_data in lines:
            line_item = gl.GLLinePlotItem(pos=np.array(line_data), color=grid_color, width=1, antialias=True)
            self.scatter_widget.addItem(line_item)

    def create_cube(self, width, height, depth, y_translation=0):
        verts = np.array([
            [width / 2, height / 2 + y_translation, depth / 2],
            [width / 2, -height / 2 + y_translation, depth / 2],
            [-width / 2, -height / 2 + y_translation, depth / 2],
            [-width / 2, height / 2 + y_translation, depth / 2],
            [width / 2, height / 2 + y_translation, -depth / 2],
            [width / 2, -height / 2 + y_translation, -depth / 2],
            [-width / 2, -height / 2 + y_translation, -depth / 2],
            [-width / 2, height / 2 + y_translation, -depth / 2]
        ])
        edges = np.array(
            [[0, 1], [1, 2], [2, 3], [3, 0], [4, 5], [5, 6], [6, 7], [7, 4], [0, 4], [1, 5], [2, 6], [3, 7]])
        cube_lines = [
            gl.GLLinePlotItem(pos=np.array([verts[edge[0]], verts[edge[1]]], dtype=np.float32), color=(1, 0, 0, 1),
                              width=2, antialias=True) for edge in edges]
        return cube_lines

    def update_scatter_plot_with_colors(self, points_with_ids, size=2):
        """
        Update the scatter plot with different colors based on target IDs.
        """
        points = points_with_ids[:, :3]  # X, Y, Z coordinates
        target_ids = points_with_ids[:, 4].astype(int)  # Extract the target IDs

        # Add the radar location
        radar_point = np.array([0, 0, 0])
        radar_id = 888
        points = np.vstack([points, radar_point])  # Append radar coordinate
        target_ids = np.append(target_ids, radar_id)  # Append radar target_id

        # Define color mapping for different target IDs
        color_map = {
            253: (1.0, 0.0, 0.0, 1.0),  # Red for SNR too weak
            254: (1.0, 0.0, 0.0, 1.0),  # Blue for points outside boundary
            255: (0.5, 0.0, 0.0, 1.0),  # Gray for noise points
            888: (1.0, 1.0, 1.0, 0.5),  # White for radar location, half-transparent
        }

        # Default color for valid points (green)
        colors = np.array([
            (0.0, 1.0, 0.0, 1.0) if target_id <= 252 else color_map.get(target_id, (1.0, 1.0, 1.0, 1.0))
            for target_id in target_ids
        ], dtype=np.float32)

        # Update the scatter plot with the points and colors
        self.scatter.setData(pos=points, color=colors, size=size)

    def update_fall_indicator(self, detected_activity):
        """
        Updates the fall indicator based on the detected activity.
        """
        if detected_activity == "Falling":
            self.fall_indicator.setText("FALL DETECTED")
            self.fall_indicator.setStyleSheet("QLabel { background-color: red; font-size: 18pt; }")
        else:
            self.fall_indicator.setText(detected_activity)
            self.fall_indicator.setStyleSheet("QLabel { background-color: green; font-size: 18pt; }")

    # Method to reset the fall indicator after a certain time
    def reset_fall_indicator(self):
        self.fall_indicator.setText("Monitoring...")
        self.fall_indicator.setStyleSheet("QLabel { background-color: green; font-size: 18pt; }")

