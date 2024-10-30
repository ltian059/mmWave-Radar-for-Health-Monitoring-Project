import os

import torch
import pickle
import numpy as np

data_path = "./8_pkl_data_1031-800-600-600/input.pkl"
model_path = "/home/uottawa/epoch155.pt"  # 修改为您的模型文件路径


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = torch.load(model_path, map_location=device)
model = model.to(device)
model.eval()

with open(data_path, "rb") as f:
    data = pickle.load(f)



input_data = torch.tensor(data).float().to(device)

batch_size = 128
output_batches = []

with torch.no_grad():
    for i in range(0, input_data.size(0), batch_size):
        batch = input_data[i:i + batch_size]  # Select batch
        output = model(batch)  # Run inference
        output_batches.append(output.cpu())  # Move to CPU and store the result

# Concatenate all batch outputs
readable_output = torch.cat(output_batches, dim=0)

output_file_path = "model_output.csv"
np.savetxt(output_file_path, readable_output, delimiter=",")

print(f"模型输出已保存到 {output_file_path}")
