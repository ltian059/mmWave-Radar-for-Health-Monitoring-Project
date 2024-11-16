import random
import numpy as np
import pandas as pd
import os
import time
import torch
import csv
import Configuration

output_csv_file_path = Configuration.output_csv_file_path

byteBuffer = np.zeros(2**15,dtype = 'uint8')
byteBufferLength = 0

previous_pc = "empty"
previous_df = pd.DataFrame()
average_zs = []
model_output_window = []
current_output_index = 0

alarm_trigger = False
window = 20
current_window_idx = 0
all_data_frame = []
fall_df = pd.DataFrame(columns = ['detected_falls_idx'])

def readAndParseData14xx(Dataport, configParameters, flush_interval=1000):
    global byteBuffer, byteBufferLength, frame_timestamp

    # Constants
    OBJ_STRUCT_SIZE_BYTES = 12
    BYTE_VEC_ACC_MAX_SIZE = 2 ** 15
    MMWDEMO_OUTPUT_MSG_COMPRESSED_POINTS = 1020
    MMWDEMO_OUTPUT_MSG_TRACKERPROC_3D_TARGET_LIST = 1010
    MMWDEMO_OUTPUT_MSG_TRACKERPROC_TARGET_INDEX = 1011
    MMWDEMO_OUTPUT_MSG_TRACKERPROC_TARGET_HEIGHT = 1012
    MMWDEMO_OUTPUT_MSG_PRESCENCE_INDICATION = 1021

    maxBufferSize = 2 ** 15
    magicWord = [2, 1, 4, 3, 6, 5, 8, 7]

    # Initialize variables
    magicOK = 0  # Checks if magic number has been read
    dataOK = 0  # Checks if the data has been read correctly
    frameNumber = 0
    detObj = {}

    readBuffer = Dataport.read(Dataport.in_waiting)
    byteVec = np.frombuffer(readBuffer, dtype='uint8')
    byteCount = len(byteVec)

    # Check that the buffer is not full, and then add the data to the buffer
    if (byteBufferLength + byteCount) < maxBufferSize:
        byteBuffer[byteBufferLength:byteBufferLength + byteCount] = byteVec
        # byteBuffer[byteBufferLength:byteBufferLength + byteCount] = byteVec[:byteCount]
        byteBufferLength = byteBufferLength + byteCount

    # Check that the buffer has some data
    if byteBufferLength > 16:
        # Check for all possible locations of the magic word
        possibleLocs = np.where(byteBuffer == magicWord[0])[0]

        # Confirm that is the beginning of the magic word and store the index in startIdx
        startIdx = []
        for loc in possibleLocs:
            check = byteBuffer[loc:loc + 8]
            if np.all(check == magicWord):
                startIdx.append(loc)

        # Check that startIdx is not empty
        if startIdx:
            # Remove the data before the first start index
            if startIdx[0] > 0 and startIdx[0] < byteBufferLength:
                byteBuffer[:byteBufferLength - startIdx[0]] = byteBuffer[startIdx[0]:byteBufferLength]
                byteBuffer[byteBufferLength - startIdx[0]:] = np.zeros(len(byteBuffer[byteBufferLength - startIdx[0]:]),
                                                                       dtype='uint8')  # uint8? ----Han
                byteBufferLength = byteBufferLength - startIdx[0]

            # Check that there have no errors with the byte buffer length
            if byteBufferLength < 0:
                byteBufferLength = 0

            # word array to convert 4 bytes to a 32 bit number
            word = [1, 2 ** 8, 2 ** 16, 2 ** 24]

            # Read the total packet length
            totalPacketLen = np.matmul(byteBuffer[12:12 + 4], word)

            # Check that all the packet has been read
            if (byteBufferLength >= totalPacketLen) and (byteBufferLength != 0):
                magicOK = 1

    # If magicOK is equal to 1 then process the message
    if magicOK:
        # word array to convert 4 bytes to a 32 bit number
        word = [1, 2 ** 8, 2 ** 16, 2 ** 24]

        # Initialize the pointer index
        idX = 0

        # Read the header
        magicNumber = byteBuffer[idX:idX + 8]
        idX += 8
        version = format(np.matmul(byteBuffer[idX:idX + 4], word), 'x')
        idX += 4
        totalPacketLen = np.matmul(byteBuffer[idX:idX + 4], word)
        idX += 4
        platform = format(np.matmul(byteBuffer[idX:idX + 4], word), 'x')
        idX += 4
        frameNumber = np.matmul(byteBuffer[idX:idX + 4], word)
        idX += 4
        timeCpuCycles = np.matmul(byteBuffer[idX:idX + 4], word)
        idX += 4
        numDetectedObj = np.matmul(byteBuffer[idX:idX + 4], word)
        idX += 4
        numTLVs = np.matmul(byteBuffer[idX:idX + 4], word)
        idX += 4

        subFrameNumber = np.matmul(byteBuffer[idX:idX + 4], word)
        idX += 4

        # Get the current timestamp
        frame_timestamp = time.time()

        # Convert the timestamp to a structured time object
        time_struct = time.localtime(frame_timestamp)

        # Extract milliseconds from the fractional part of the timestamp
        milliseconds = int((frame_timestamp - int(frame_timestamp)) * 1000)

        # Format the time string
        frame_timestamp = time.strftime("%H:%M:%S", time_struct) + f":{milliseconds:03d}"

        # Read the TLV messages
        for tlvIdx in range(numTLVs):
            word = [1, 2 ** 8, 2 ** 16, 2 ** 24]

            # Check the header of the TLV message
            original = byteBuffer[idX:idX + 4]
            tlv_type = np.matmul(byteBuffer[idX:idX + 4], word)
            idX += 4
            tlv_length = np.matmul(byteBuffer[idX:idX + 4], word)
            idX += 4
            # print('Original:', original)
            # print('TLV type:', tlv_type)

            # Read the data depending on the TLV message

            if tlv_type == MMWDEMO_OUTPUT_MSG_COMPRESSED_POINTS:
                # print('we are in 1020')

                # idX += tlv_length
                pointCloud = []

                elevationUnit = np.frombuffer(byteBuffer[idX:idX + 4], dtype=np.float32)[0]
                idX += 4
                azimuthUnit = np.frombuffer(byteBuffer[idX:idX + 4], dtype=np.float32)[0]
                idX += 4
                dopplerUnit = np.frombuffer(byteBuffer[idX:idX + 4], dtype=np.float32)[0]
                idX += 4
                rangeUnit = np.frombuffer(byteBuffer[idX:idX + 4], dtype=np.float32)[0]
                idX += 4
                snrUnit = np.frombuffer(byteBuffer[idX:idX + 4], dtype=np.float32)[0]
                idX += 4

                numPoints = (tlv_length - 20) // 8

                for _ in range(numPoints):
                    elevation = np.frombuffer(byteBuffer[idX:idX + 1], dtype=np.int8)[0] * elevationUnit
                    idX += 1
                    azimuth = np.frombuffer(byteBuffer[idX:idX + 1], dtype=np.int8)[0] * azimuthUnit
                    idX += 1
                    doppler = np.frombuffer(byteBuffer[idX:idX + 2], dtype=np.int16)[0] * dopplerUnit
                    idX += 2
                    rangeVal = np.frombuffer(byteBuffer[idX:idX + 2], dtype=np.uint16)[0] * rangeUnit
                    idX += 2
                    snr = np.frombuffer(byteBuffer[idX:idX + 2], dtype=np.uint16)[0] * snrUnit
                    idX += 2

                    # Convert spherical to Cartesian coordinates
                    x = rangeVal * np.cos(elevation) * np.sin(azimuth)
                    y = rangeVal * np.cos(elevation) * np.cos(azimuth)
                    z = rangeVal * np.sin(elevation)

                    pointCloud.append([x, y, z, doppler, snr])

                detObj['pointCloud'] = pointCloud
                dataOK = 1

            elif tlv_type == MMWDEMO_OUTPUT_MSG_TRACKERPROC_3D_TARGET_LIST:  # problem
                # print('we are in 1010')

                # idX += tlv_length

                # print("tlv_length", tlv_length)

                # orig_idX = idX

                targetList = []

                # numTargets = (tlv_length - 8) // (4 + 4*13 + 4*16 + 4*2) + 1   # can be solved by + 1 at the end but don't know why
                numTargets = tlv_length // (4 * 12 + 4 * 16)

                for _ in range(numTargets):
                    tid = np.frombuffer(byteBuffer[idX:idX + 4], dtype=np.uint32)[0]
                    idX += 4
                    posX = np.frombuffer(byteBuffer[idX:idX + 4], dtype=np.float32)[0]
                    idX += 4
                    posY = np.frombuffer(byteBuffer[idX:idX + 4], dtype=np.float32)[0]
                    idX += 4
                    posZ = np.frombuffer(byteBuffer[idX:idX + 4], dtype=np.float32)[0]
                    idX += 4
                    velX = np.frombuffer(byteBuffer[idX:idX + 4], dtype=np.float32)[0]
                    idX += 4
                    velY = np.frombuffer(byteBuffer[idX:idX + 4], dtype=np.float32)[0]
                    idX += 4
                    velZ = np.frombuffer(byteBuffer[idX:idX + 4], dtype=np.float32)[0]
                    idX += 4
                    accX = np.frombuffer(byteBuffer[idX:idX + 4], dtype=np.float32)[0]
                    idX += 4
                    accY = np.frombuffer(byteBuffer[idX:idX + 4], dtype=np.float32)[0]
                    idX += 4
                    accZ = np.frombuffer(byteBuffer[idX:idX + 4], dtype=np.float32)[0]
                    idX += 4
                    EC = np.frombuffer(byteBuffer[idX:idX + 64], dtype=np.float32).reshape(4, 4)
                    idX += 64
                    G = np.frombuffer(byteBuffer[idX:idX + 4], dtype=np.float32)[0]
                    idX += 4
                    confidenceLevel = np.frombuffer(byteBuffer[idX:idX + 4], dtype=np.float32)[0]
                    idX += 4

                    targetList.append({
                        'tid': tid,
                        'posX': posX,
                        'posY': posY,
                        'posZ': posZ,
                        'velX': velX,
                        'velY': velY,
                        'velZ': velZ,
                        'accX': accX,
                        'accY': accY,
                        'accZ': accZ,
                        'EC': EC,
                        'G': G,
                        'confidenceLevel': confidenceLevel
                    })

                detObj['targetList'] = targetList
                # diff = idX - orig_idX
                # print("diff", diff)

            elif tlv_type == MMWDEMO_OUTPUT_MSG_TRACKERPROC_TARGET_INDEX:  # problem - solved
                # print('we are in 1011')
                # print("tlvlength:", tlv_length)
                # orig_idX = idX
                # idX += tlv_length
                targetIndices = []

                numPoints = tlv_length

                for _ in range(numPoints):
                    targetID = np.frombuffer(byteBuffer[idX:idX + 1], dtype=np.uint8)[0]
                    idX += 1
                    targetIndices.append(targetID)

                detObj['targetIndices'] = targetIndices
                # diff = idX - orig_idX
                # print("diff:",diff)

            # else:
            #     idX += tlv_length

            elif tlv_type == MMWDEMO_OUTPUT_MSG_PRESCENCE_INDICATION:
                idX += tlv_length
                # idX += 12 # 8+4

            elif tlv_type == MMWDEMO_OUTPUT_MSG_TRACKERPROC_TARGET_HEIGHT:

                # problematic code.
                # targetHeight = []

                # numTargets = (tlv_length) // (1+4+4)
                # for _ in range(numTargets):
                #     targetID = np.frombuffer(byteBuffer[idX:idX+1], dtype=np.uint8)[0]
                #     idX += 1
                #     maxZ = np.frombuffer(byteBuffer[idX:idX+4], dtype=np.float32)[0]
                #     idX += 4
                #     minZ = np.frombuffer(byteBuffer[idX:idX+4], dtype=np.float32)[0]
                #     idX += 4

                #     targetHeight.append({
                #         'targetID': targetID,
                #         'maxZ': maxZ,
                #         'minZ': minZ
                #     })

                # detObj['targetHeight'] = targetHeight
                idX += tlv_length

        # Remove already processed data
        if idX > 0 and byteBufferLength > idX:  # why "byteBufferLength > idX"? -----Xilai
            shiftSize = totalPacketLen
            byteBuffer[:byteBufferLength - shiftSize] = byteBuffer[shiftSize:byteBufferLength]
            byteBuffer[byteBufferLength - shiftSize:] = np.zeros(len(byteBuffer[byteBufferLength - shiftSize:]),
                                                                 dtype='uint8')
            byteBufferLength = byteBufferLength - shiftSize

            # Check that there are no errors with the buffer length
            if byteBufferLength < 0:
                byteBufferLength = 0
        return dataOK, frameNumber, detObj, frame_timestamp

    return dataOK, frameNumber, detObj, None

# Function to save DataFrame to CSV
def save_to_csv(df, file_path):
    if not df.empty:
        dir_path = os.path.dirname(file_path)
        if not os.path.exists(dir_path):
            os.makedirs(dir_path)
        # Save the DataFrame to the new CSV file
        df.to_csv(file_path, mode='a', index=False, header=not os.path.isfile(file_path))


def append_to_csv(output, file_path):
    timestamp = time.strftime("%Y-%m-%d %H:%M:%S")  # Get current time in readable format
    with open(file_path, mode='a', newline='') as file:
        writer = csv.writer(file)
        writer.writerow([timestamp, output])


def fill_frame(frame_data, target_length):
    frame_length = len(frame_data)
    if frame_length >= target_length:
        return frame_data[:target_length]

    filled_frame = frame_data.copy()
    # print("frame length:",frame_length)
    for _ in range(target_length - frame_length):
        random_index = random.randint(0, frame_length-1)
        random_point = frame_data[random_index]
        filled_frame.append(random_point)
    return filled_frame


# 6 cls withT transition removed or not:
def num_to_class(num):
    if num == 0:
        return "Falling"
    elif num == 1:
        return "Laying"
    elif num == 2:
        return "Sitting"
    elif num == 3:
        return "Stand+Walking"
    else:
        return "Transition"

processed_frame_numbers = []
def update(Dataport, configParameters, model, device, gui):
    global s, recent_predictions, alarm_trigger, window, current_window_idx, all_data_frame, fall_df
    global previous_pc, previous_df, average_zs, model_output_window, current_output_index

    try:
        dataOk, frameNumber, detObj, frame_timestamp = readAndParseData14xx(Dataport, configParameters)

        if dataOk:
            pointCloud = np.array(detObj['pointCloud'])  # (N, 5) array: X, Y, Z, velocity, snr
            targetIndices = detObj.get('targetIndices', [])  # List of target indices


            if frame_timestamp:
                pass
            else:
                frame_timestamp = "No timestamp available"
            # Create a DataFrame for the point cloud
            df = pd.DataFrame({
                'frame_timestamp': frame_timestamp,
                'Frame Number': frameNumber,
                'X': pointCloud[:, 0],
                'Y': pointCloud[:, 1],
                'Z': pointCloud[:, 2],
                'velocity': pointCloud[:, 3],
                'snr': pointCloud[:, 4],
            })

            # Log the frame number
            processed_frame_numbers.append(frameNumber)

            # Stack the X, Y, Z columns to form the point cloud shape (N, 3)
            pointCloud1 = np.vstack((df['X'], df['Y'], df['Z'], df['velocity'])).T

            # Check if previous_df is empty (first frame case)
            if previous_df.empty:
                previous_df = df

            # Check if previous_pc is empty or has no points (first frame case)
            if isinstance(previous_pc, str) and previous_pc == "empty" or previous_pc.size == 0:
                previous_pc = pointCloud1
                return

            # Store the current point cloud temporarily and update `previous_pc`
            current_pc = pointCloud1
            points = previous_pc  # Use the previous frame's points to match target IDs
            previous_pc = current_pc  # Update previous_pc with the current point cloud for the next iteration

            # Handle cases where target indices do not match point cloud length
            if len(targetIndices) != points.shape[0]:
                previous_df = df
                return

            # Save the previous frame's DataFrame with the associated target IDs
            if len(targetIndices) == len(previous_df):
                previous_df['target_id'] = targetIndices
            else:
                previous_df['target_id'] = [-1] * len(previous_df)  # Assign a default value (-1) for missing target IDs

            save_to_csv(previous_df, Configuration.csv_file_path_timestamp)  # Save previous_df to CSV
            previous_df = df  # Update for the next frame

            # Create a mask to filter out invalid target IDs (e.g., 253, 254, 255)
            mask = np.array([(0 <= x <= 255) for x in targetIndices])  # True for valid IDs

            # Apply the mask to filter out invalid points
            valid_points = points[mask]  # Filtered point cloud (N, 3)
            valid_target_ids = np.array(targetIndices)[mask]  # Corresponding valid target IDs (N,)

            # Prepare the points for processing by adding the target ID as a fourth dimension
            points_with_ids = np.hstack((valid_points, valid_target_ids[:, np.newaxis]))

            z_values = valid_points[:, 2]
            z_mean = np.mean(z_values)
            average_zs.append(z_mean)
            valid_points = valid_points.tolist()

            if len(valid_points) == 0:
                return

            filled_frame = fill_frame(valid_points,
                                      target_length=150)  # Now filled_frame is a raw Python list of points in ONE frame

            if df.empty:
                pass
            else:
                current_window_idx += 1
                all_data_frame.append(filled_frame)

            if current_window_idx == 20:
                input = torch.Tensor(all_data_frame).unsqueeze(0).to(device)
                output = model(input)
                _, predicted_label = torch.max(output, 1)
                str_class = num_to_class(predicted_label.item())

                model_output_window.append(str_class)
                current_output_index += 1

                if current_output_index == 10:
                    afterward_output = 'not assigned'
                    if sum(1 for x in model_output_window if x in ['Falling', 'Laying']) > 4:
                        afterward_output = 'Falling'
                    elif sum(1 for x in model_output_window if x == 'Sitting') > 2:
                        afterward_output = 'Sitting'
                    else:
                        afterward_output = 'Standing / \nWalking'
                    print(afterward_output)  # This prints the result
                    gui.update_indicator(afterward_output)

                    # Now append it to the CSV file along with the timestamp
                    append_to_csv(afterward_output, output_csv_file_path)

                    current_output_index = 5
                    model_output_window = model_output_window[5:]

                current_window_idx = 15
                all_data_frame = all_data_frame[5:]
                gui.update_scatter_plot_with_colors(points_with_ids)
        else:
            pass
    except Exception as e:
        print(f"Error processing frame {frameNumber}: {e}")


def close_ports(CLIport, Dataport):
    if CLIport and CLIport.is_open:
        CLIport.write('sensorStop\n'.encode())
        time.sleep(0.2)
        CLIport.write('flushCfg\n'.encode())
        time.sleep(0.2)
        CLIport.reset_input_buffer()
        CLIport.reset_output_buffer()
        CLIport.close()
    if Dataport and Dataport.is_open:
        Dataport.close()



def generate_csv_title_timestamp():
    dir_path = os.path.dirname(Configuration.csv_file_path)
    # Get current time in dd_mm_ss format
    time_str = time.strftime("%Hh_%Mm_%Ss_%d_%m_%y")
    # Extract base file name and extension
    base_name = os.path.basename(Configuration.csv_file_path)
    name, ext = os.path.splitext(base_name)
    # Create new file name with time appended
    new_file_name = f"{name}_{time_str}{ext}"
    # Construct full path for the new file
    new_file_path = os.path.join(dir_path, new_file_name)
    return new_file_path