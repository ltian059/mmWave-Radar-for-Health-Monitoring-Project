import os
import torch
import pickle
import numpy as np
import time
import sys
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler
sys.stdout.reconfigure(line_buffering=True)
"""
File: test.py
Description:
    This script performs real-time inference on incoming data files using a pre-trained PyTorch model.
    It continuously monitors a specified directory for new `.pkl` files, processes them using the model, 
    and outputs results in CSV format to an output directory. The script leverages the `watchdog` library 
    to detect new files in real-time and processes them in batches for efficiency.

Workflow:
    1. **Model and Directory Setup**:
        - Loads a pre-trained model from the specified path.
        - Ensures input and output directories exist.

    2. **File Monitoring**:
        - Watches for new `.pkl` files in the input directory.
        - Checks file stability to ensure files are fully written before processing.

    3. **Batch Processing**:
        - Processes input data in batches to optimize memory usage and model performance.
        - Performs inference with PyTorch, using either GPU or CPU based on device availability.

    4. **Output Handling**:
        - Saves the inference output as a CSV file in the output directory.
        - Keeps track of processed files to avoid duplicate processing.
        - Cleans old entries from tracking based on a defined retention period.

Usage:
    - Place the `.pkl` files in the input directory for automatic processing.
    - The script will output the inference results as CSV files in the output directory.

Dependencies:
    - torch: For model loading and inference.
    - watchdog: For monitoring the input directory for new files.
    - pickle: For loading input data files in `.pkl` format.
    - numpy: For saving output data in CSV format.

Notes:
    - Set `model_path` to the path of the trained model file (`epoch155.pt`).
    - Ensure `input_folder` and `output_folder` are correctly specified.

Author:
    Li Tian
"""

# Define paths
model_path = "./epoch155.pt"  # Modify to your model file path
input_folder = "./input_data"  # Folder where data files will arrive
output_folder = "./output"

# Check device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if device.type == "cuda":
    print("Using GPU for inference", flush=True)
else:
    print("Using CPU for inference", flush=True)

# Load the model
model = torch.load(model_path, map_location=device)
model = model.to(device)
model.eval()
# Ensure the output directory exists

os.makedirs(output_folder, exist_ok=True)
os.makedirs(input_folder, exist_ok=True)  # Ensure input folder exists

# Set batch size and prepare for batch processing
batch_size = 64


def is_file_stable(file_path, wait_time=0.1):
    """
       Checks if a file has been fully written and is stable by comparing its size over a short interval.

       Args:
           file_path (str): Path to the file being checked.
           wait_time (float): Time to wait between size checks, in seconds. Default is 0.1 seconds.

       Returns:
           bool: True if the file size is stable (has not changed during the interval), False otherwise.

       Purpose:
           Prevents processing of incomplete or partially written files by waiting until the file size
           remains constant for a short duration.
   """
    size1 = os.path.getsize(file_path)
    time.sleep(wait_time)
    size2 = os.path.getsize(file_path)
    return size1 == size2


def process_file(data_path):
    """
        Processes an incoming data file by loading, performing inference, and saving the output.

        Args:
            data_path (str): Path to the .pkl data file to be processed.

        Process Overview:
            1. Ensures file is fully written and stable before processing.
            2. Loads the file using pickle and converts it into a tensor for inference.
            3. Performs batch-wise inference on the loaded data without gradients to save memory.
            4. Concatenates the batched outputs, converts them to CPU, and saves them in CSV format.

        Notes:
            Uses torch.no_grad() to disable gradient calculation for efficiency.
            Batch processing is done to manage memory usage and improve efficiency on large inputs.
    """
    # Wait until the file has a non-zero size to avoid EOFError
    while os.path.getsize(data_path) == 0:
        print(f"Waiting for file {data_path} to be fully written...")
        time.sleep(0.1)  # Check every 100 milliseconds

    while not is_file_stable(data_path):
        print(f"Waiting for file {data_path} to be fully written...")

    with open(data_path, "rb") as f:
        data = pickle.load(f)
    print(f"data loaded: File name{data_path}")
    input_data = torch.tensor(data).float().to(device)
    output_batches = []

    # Inference without gradients
    with torch.no_grad():
        for i in range(0, input_data.size(0), batch_size):
            batch = input_data[i:i + batch_size]  # Select batch
            output = model(batch)  # Run inference
            output_batches.append(output.cpu())  # Move to CPU and store the result

    # Concatenate and save output
    readable_output = torch.cat(output_batches, dim=0)
    output_file_path = os.path.join(output_folder, f"output_{os.path.basename(data_path).split('.')[0]}.csv")
    np.savetxt(output_file_path, readable_output, delimiter=",")
    print(f"Processed {data_path} and saved output to {output_file_path}")


last_processed = {}
RETENTION_PERIOD = 24 * 60 * 60


def clean_old_files():
    """
        Removes entries in the `last_processed` dictionary if their modification times exceed
        a defined retention period (24 hours).

        Purpose:
            Keeps the `last_processed` dictionary clean and removes obsolete file entries that
            are no longer needed. This function runs after each file processing to ensure only
            recent files are retained, avoiding memory bloat.

        Notes:
            `RETENTION_PERIOD` is set to 24 hours, so any file entry older than this will be removed.
    """
    # Remove entries from `last_processed` older than the retention period
    current_time = time.time()
    to_remove = [filename for filename, mod_time in last_processed.items()
                 if current_time - mod_time > RETENTION_PERIOD]
    for filename in to_remove:
        del last_processed[filename]


# Event handler for new files
class NewFileHandler(FileSystemEventHandler):
    """
      A custom handler for new file events in the specified directory. This handler is triggered
      whenever a new file is created, enabling immediate processing if it's a .pkl file.

      Event:
          `on_created`: Called when a new file or directory is created.

      Attributes:
          - None; uses the `last_processed` dictionary and external `process_file` function.

      Methods:
          on_created(event): Checks if the new file is a .pkl file and has a newer modification time.
          If so, it processes the file and updates `last_processed`.

      Purpose:
          Enables real-time processing of new .pkl files in the watched directory.
  """
    def on_created(self, event):
        # Check if the new file is a .pkl file
        if event.is_directory:
            return
        filename = os.path.basename(event.src_path)
        if filename.endswith(".pkl"):
            modified_time = os.path.getmtime(event.src_path)
            if filename not in last_processed or last_processed[filename] < modified_time:
                print(f"Detected new file: {filename}, processing immediately.")
                process_file(event.src_path)
                last_processed[filename] = modified_time # Update `last_processed` timestamp
                clean_old_files() # Remove outdated entries


# Set up the watchdog observer
observer = Observer()
event_handler = NewFileHandler()
observer.schedule(event_handler, path=input_folder, recursive=False)

# Start the observer in the background
observer.start()
print("Monitoring directory for new data...")

try:
    observer.join()
except KeyboardInterrupt:
    observer.stop()

observer.join()
