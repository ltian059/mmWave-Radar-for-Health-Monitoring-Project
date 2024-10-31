import os
import torch
import pickle
import numpy as np
import time
# Define paths
model_path = "./epoch155.pt"  # Modify to your model file path
input_folder = "./input_data" # Folder where data files will arrive
output_folder = "./output"

# Check device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if device.type == "cuda":
    print("Using GPU for inference")
else:
    print("Using CPU for inference")

# Load the model
model = torch.load(model_path, map_location=device)
model = model.to(device)
model.eval()
# Ensure the output directory exists

os.makedirs(output_folder, exist_ok=True)
os.makedirs(input_folder, exist_ok=True)  # Ensure input folder exists

# Set batch size and prepare for batch processing
batch_size = 64

def process_file(data_path):
    # Load input data
    with open(data_path, "rb") as f:
        data = pickle.load(f)
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
while True:
    # Check for new `.pkl` files in the input folder
    current_time = time.time()
    last_processed = {k: v for k, v in last_processed.items() if current_time - v < RETENTION_PERIOD}
    for filename in os.listdir(input_folder):
        if filename.endswith(".pkl"):
            data_path = os.path.join(input_folder, filename)
            modified_time = os.path.getmtime(data_path)
            if filename not in last_processed or last_processed[filename] < modified_time:
                process_file(data_path)
                last_processed[filename] = modified_time

    # Sleep to avoid excessive polling
    time.sleep(5)