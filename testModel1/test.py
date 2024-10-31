import os
import torch
import pickle
import numpy as np

# Define paths
data_path = "./8_pkl_data_1031-800-600-600/input.pkl"
model_path = "./epoch155.pt"  # Modify to your model file path
output_folder = "./output"
output_file_path = os.path.join(output_folder, "model_output.csv")

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

# Load the input data
with open(data_path, "rb") as f:
    data = pickle.load(f)

# Convert data to tensor and move to device
input_data = torch.tensor(data).float().to(device)

# Ensure the output directory exists
os.makedirs(output_folder, exist_ok=True)

# Set batch size and prepare for batch processing
batch_size = 128
output_batches = []

# Inference without gradients
with torch.no_grad():
    for i in range(0, input_data.size(0), batch_size):
        batch = input_data[i:i + batch_size]  # Select batch
        output = model(batch)  # Run inference
        output_batches.append(output.cpu())  # Move to CPU and store the result

# Concatenate all batch outputs
readable_output = torch.cat(output_batches, dim=0)

# Save output to CSV
np.savetxt(output_file_path, readable_output, delimiter=",")

print(f"The output of the model has been saved to {output_file_path}")
