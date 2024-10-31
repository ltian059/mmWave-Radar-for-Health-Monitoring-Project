# Main script: test.py
## Description:
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

**Usage**:
    - Place the `.pkl` files in the input directory for automatic processing.
    - The script will output the inference results as CSV files in the output directory.

**Dependencies**:
    - torch: For model loading and inference.
    - watchdog: For monitoring the input directory for new files.
    - pickle: For loading input data files in `.pkl` format.
    - numpy: For saving output data in CSV format.

**Notes**:
    - Set `model_path` to the path of the trained model file (`epoch155.pt`).
    - Ensure `input_folder` and `output_folder` are correctly specified.

**Author**:
    Li Tian
