import os
import torch
import pickle
import numpy as np
import time
import sys
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler
sys.stdout.reconfigure(line_buffering=True)

# Define paths
model_path = "./epoch155.pt"  # Modify to your model file path
input_folder = "./input_data" # Folder where data files will arrive
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
    size1 = os.path.getsize(file_path)
    time.sleep(wait_time)
    size2 = os.path.getsize(file_path)
    return size1 == size2

def process_file(data_path):
    # Load input data
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
    # Remove entries from `last_processed` older than the retention period
    current_time = time.time()
    to_remove = [filename for filename, mod_time in last_processed.items()
                 if current_time - mod_time > RETENTION_PERIOD]
    for filename in to_remove:
        del last_processed[filename]

# Event handler for new files
class NewFileHandler(FileSystemEventHandler):
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
                last_processed[filename] = modified_time
                clean_old_files()

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