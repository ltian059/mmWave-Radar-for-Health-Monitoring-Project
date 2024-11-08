import atexit

import serial
import time
import numpy as np
import pyqtgraph as pg
from pyqtgraph.Qt import QtGui
#import matplotlib.pyplot as plt
import serial
import time
import numpy as np
import pyqtgraph as pg
from pyqtgraph.Qt import QtGui
import sys
from PyQt5 import QtGui, QtWidgets, QtCore
from pyqtgraph.opengl import GLViewWidget, GLScatterPlotItem
import csv
import pandas as pd
import argparse, os
import random as rn
import tensorflow as tf
from keras import backend as K
from keras import optimizers
from keras.layers import Input, Dense, Flatten, Lambda, Concatenate, Reshape, \
    TimeDistributed, LSTM, RepeatVector, SimpleRNN, Activation
from keras.models import Model, load_model
from keras.callbacks import TensorBoard
from keras.losses import MeanSquaredError
mse = MeanSquaredError()
# from keras.losses import mse
from keras.utils import plot_model
#from scipy.signal import find_peaks
#from sklearn.metrics import confusion_matrix
import pandas as pd
from tensorflow.keras import layers
from scipy.signal import butter, filtfilt
import glob
import struct

from keras.layers import Layer
from tensorflow.python.framework.ops import disable_eager_execution

disable_eager_execution()

class data_preproc:
    def __init__(self):
        self.frames_per_pattern = 20
        self.points_per_frame = 64
        self.features_per_point = 4
        self.split_ratio = 0.8
        tilt_angle = -10.0
        self.height = 2
        self.rotation_matrix = np.array([[1.0, 0.0, 0.0],
                                         [0.0, np.cos(np.deg2rad(tilt_angle)), -np.sin(np.deg2rad(tilt_angle))],
                                         [0.0, np.sin(np.deg2rad(tilt_angle)), np.cos(np.deg2rad(tilt_angle))]])

    def load_csv(self, data_frame, anomaly=False):
        centroidX_his = []
        centroidY_his = []
        centroidZ_his = []
        total_processed_pattern = []

        df = data_frame
    
        # Number of frames to process at once
        frames_batch_size = 20

        # Get unique frame numbers
        unique_frames = df['Frame Number'].unique()

        num_complete_batches = len(unique_frames) // frames_batch_size
        # print ('num_complete_batches', num_complete_batches)
        # print ('len(unique_frames)', len(unique_frames))
        # print ('frames_batch_size', frames_batch_size)
        # os.system("Pause")
        # Loop over the complete batches
        for batch_num in range(num_complete_batches):
            start = batch_num * frames_batch_size
            end = start + frames_batch_size
            frame_numbers = unique_frames[start:end]
            processed_pattern = []  # This will hold all processed frames in the current batch
            # Take the first frame number
            # first_frame_number = frame_numbers[0]
        
            # # Select all rows for the first frame
            # frame_data = df[df['Frame Number'] == first_frame_number]
            # centroid = frame_data[['X', 'Y', 'Z']].mean().to_numpy()
            # # print('centroid',centroid)
            # # os.system('pause')
            # centroidx = centroid[0]
            # centroidy = centroid[1]
            # centroidz = centroid[2]
            # # print('centroid x', centroidx)
            # # os.system('Pause')
            # results      = np.matmul(self.rotation_matrix, np.array([centroidx,centroidy,centroidz]))
            # centroidx    = results[0]
            # centroidy    = results[1]
            # centroidz    = results[2] + self.height
            
            # centroidX_his.append(centroidx)
            # centroidY_his.append(centroidy)
            # centroidZ_his.append(centroidz)
            # # print('centroid x', centroidx)
            # # os.system('Pause')
            for frame_number in frame_numbers:
                group = df[df['Frame Number'] == frame_number]

                if len(group) > self.points_per_frame:
                    continue
                # print('max poioint per frame', self.points_per_frame)
                # print('lenghth group', len(group))
                # os.system('Pause')   # Skip frames with insufficient points

                centroid = group[['X', 'Y', 'Z']].mean().to_numpy()
                # print('centroid',centroid)
                # os.system('pause')
                centroidx = centroid[0]
                centroidy = centroid[1]
                centroidz = centroid[2]
                # print('centroid x', centroidx)
                # os.system('Pause')
                results      = np.matmul(self.rotation_matrix, np.array([centroidx,centroidy,centroidz]))
                centroidx    = results[0]
                centroidy    = results[1]
                centroidz    = results[2] + self.height
                
                centroidX_his.append(centroidx)
                centroidY_his.append(centroidy)
                centroidZ_his.append(centroidz)
                # print('centroid x', centroidx)
                # os.system('Pause')

                
                processed_frame = []
                for _, row in group.iterrows():
                    # Apply rotation and adjust for height
                    point = row[['X', 'Y', 'Z']].to_numpy()
                    # print('Point', point)
                    # os.system('Pause')
                    rotated_point = np.matmul(self.rotation_matrix, row[['X', 'Y', 'Z']].to_numpy())
                    pointX, pointY, pointZ = rotated_point + np.array([0, 0, self.height])
                    
                    # Calculate deltas
                    delta_x = pointX - centroidx
                    delta_y = pointY - centroidy
                    delta_z = pointZ #- centroidz + self.height  # Adjusting centroid for height as well
                    delta_D = row['velocity']  
                    
                    # Form the feature vector
                    feature_vector = [delta_x, delta_y, delta_z, delta_D]
                    processed_frame.append(feature_vector)
                    #processed_frame.to_numpy()
                    # print('processed frame', processed_frame[0])
                    # os.system('Pause')
                #processed_pattern.append(processed_frame)
                
                processed_pattern.append(processed_frame)
                # print('Processed pattern =', len(processed_pattern[0]))
                # os.system('Pause')
            if len(processed_pattern) == frames_batch_size:
                processed_pattern_oversampled = self.proposed_oversampling(processed_pattern)
                total_processed_pattern.append(processed_pattern_oversampled)
        # print('total_processed_pattern shape ==> ', len(total_processed_pattern[0]))
        # print('total_processed_pattern ==> \n ', total_processed_pattern)
        total_processed_pattern_np = np.array(total_processed_pattern)
        # print('total_processed_pattern_np shape', total_processed_pattern_np.shape)
        # print('total_processed_pattern_np', total_processed_pattern_np)
        #os.system('Pause')
        # Split data into training and testing sets
        split_idx = int(total_processed_pattern_np.shape[0] * self.split_ratio)
        train_data = total_processed_pattern_np[:split_idx]
        test_data = total_processed_pattern_np[split_idx:]

        if anomaly == False:
            print("INFO: Total normal motion pattern data shape: " + str(total_processed_pattern_np.shape))
            print("INFO: Training motion pattern data shape" + str(train_data.shape))
            print("INFO: Testing motion pattern data shape" + str(test_data.shape))
            # Return training and testing data along with centroid histories for normal dataset
            return train_data, test_data, centroidZ_his
        else:
            # Return processed pattern and centroid histories for anomaly dataset
            print("INFO: Total inference motion pattern data shape: " + str(total_processed_pattern_np.shape))
            return total_processed_pattern_np,  centroidZ_his
    
    def proposed_oversampling(self, processed_pointcloud):
        # # Check the input
        # point_list = []
        # for frame in processed_pointcloud:
        #     point_list.extend(frame)
        # point_list_np  = np.array(point_list)
        # assert (point_list_np.shape[-1] == self.features_per_point), ("ERROR: Input processed_pointcloud has different feature length rather than %s!" %(self.features_per_point))

        # Do the data oversampling
        processed_pointcloud_oversampled = []
        for frame in processed_pointcloud:
            frame_np = np.array(frame)
            # Check if it's empty frame
            N = self.points_per_frame
            M = frame_np.shape[0]
            assert (M != 0), "ERROR: empty frame detected!"
            # Rescale and padding
            mean        = np.mean(frame_np, axis=0)
            sigma       = np.std(frame_np, axis=0)
            frame_np    = np.sqrt(N/M)*frame_np + mean - np.sqrt(N/M)*mean # Rescale
            frame_oversampled = frame_np.tolist()
            frame_oversampled.extend([mean]*(N-M)) # Padding with mean
            # # Check if mean and sigma keeps the same
            # new_mean    = np.mean(np.array(frame_oversampled), axis=0)
            # new_sigma   = np.std(np.array(frame_oversampled), axis=0)
            # assert np.sum(np.abs(new_mean-mean))<1e-5, ("ERROR: Mean rescale and padding error!")
            # assert np.sum(np.abs(new_sigma-sigma))<1e-5, ("ERROR: Sigma rescale and padding error!")
            processed_pointcloud_oversampled.append(frame_oversampled)

        processed_pointcloud_oversampled_np = np.array(processed_pointcloud_oversampled)
        
        assert (processed_pointcloud_oversampled_np.shape[-2] == self.points_per_frame), ("ERROR: The new_frame_data has different number of points per frame rather than %s!" %(self.points_per_frame))    
        assert (processed_pointcloud_oversampled_np.shape[-1] == self.features_per_point), ("ERROR: The new_frame_data has different feature length rather than %s!" %(self.features_per_point))    

        return processed_pointcloud_oversampled_np


class SamplingLayer(layers.Layer):
    """Sampling layer for Variational Autoencoder"""
    def call(self, inputs):
        z_mean, z_log_var = inputs
        batch = tf.shape(z_mean)[0]
        dim1 = tf.shape(z_mean)[1]  # Additional dimensions if present
        dim2 = tf.shape(z_mean)[2]  # You adjust this based on your specific needs
        # Adjust the shape of epsilon based on the shape of your z_mean and z_log_var
        epsilon = tf.keras.backend.random_normal(shape=(batch, dim1, dim2))
        return z_mean + tf.exp(0.5 * z_log_var) * epsilon

class autoencoder_mdl:
    def __init__(self, model_dir):
        self.model_dir = model_dir

    # Variational Recurrent Autoencoder (HVRAE)
    def HVRAE_train(self, train_data, test_data):
        # In one motion pattern we have
        n_frames       = 20
        n_points       = 64
        n_features     = 4
        
        # Dimension is going down for encoding. Decoding is just a reflection of encoding.
        n_intermidiate    = 64
        n_latentdim       = 16
        
        # Define input
        inputs                  = Input(shape=(n_frames, n_points, n_features))
        input_flatten           = TimeDistributed(Flatten(None))(inputs)

        # VAE: q(z|X). Input: motion pattern. Output: mean and log(sigma^2) for q(z|X).
        input_flatten           = TimeDistributed(Dense(n_intermidiate, activation='tanh'))(input_flatten)
        Z_mean                  = TimeDistributed(Dense(n_latentdim, activation=None), name='qzx_mean')(input_flatten)
        Z_log_var               = TimeDistributed(Dense(n_latentdim, activation=None), name='qzx_log_var')(input_flatten)
        # def sampling(args): # Instead of sampling from Q(z|X), sample epsilon = N(0,I), z = z_mean + sqrt(var) * epsilon
        #     Z_mean, Z_log_var   = args
        #     batch_size          = K.shape(Z_mean)[0]
        #     n_frames            = K.int_shape(Z_mean)[1]
        #     n_latentdim         = K.int_shape(Z_mean)[2]
        #     # For reproducibility, we set the seed=37
        #     epsilon             = K.random_normal(shape=(batch_size, n_frames, n_latentdim), mean=0., stddev=1.0, seed=None)
        #     Z                   = Z_mean + K.exp(0.5*Z_log_var) * epsilon # The reparameterization trick
        #     return  Z
        # # VAE: sampling z ~ q(z|X) using reparameterization trick. Output: samples of z.
        Z = SamplingLayer()([Z_mean, Z_log_var])

        # RNN Autoencoder. Output: reconstructed z.
        encoder_feature         = SimpleRNN(n_latentdim, activation='tanh', return_sequences=False)(Z)
        decoder_feature         = RepeatVector(n_frames)(encoder_feature)
        decoder_feature         = SimpleRNN(n_latentdim, activation='tanh', return_sequences=True)(decoder_feature)
        decoder_feature         = Lambda(lambda x: tf.reverse(x, axis=[-2]))(decoder_feature)

        # VAE: p(X|z). Output: mean and log(sigma^2) for p(X|z).
        X_latent                = TimeDistributed(Dense(n_intermidiate, activation='tanh'))(decoder_feature)
        pXz_mean                = TimeDistributed(Dense(n_features, activation=None))(X_latent)
        pXz_logvar              = TimeDistributed(Dense(n_features, activation=None))(X_latent)

        # Reshape the output. Output: (n_frames, n_points, n_features*2).
        # In each frame, every point has a corresponding mean vector with length of n_features and a log(sigma^2) vector with length of n_features.
        pXz                     = Concatenate()([pXz_mean, pXz_logvar])
        pXz                     = TimeDistributed(RepeatVector(n_points))(pXz)
        outputs                 = TimeDistributed(Reshape((n_points, n_features*2)))(pXz)

        # Build the model
        self.HVRAE_mdl = Model(inputs, outputs)
        print(self.HVRAE_mdl.summary())

        # Calculate HVRAE loss proposed in the paper
        def HVRAE_loss(y_true, y_pred):
            batch_size      = K.shape(y_true)[0]
            n_frames        = K.shape(y_true)[1]
            n_features      = K.shape(y_true)[-1]

            mean            = y_pred[:, :, :, :n_features]
            logvar          = y_pred[:, :, :, n_features:]
            var             = K.exp(logvar)

            y_true_reshape  = K.reshape(y_true, (batch_size, n_frames, -1)) 
            mean            = K.reshape(mean, (batch_size, n_frames, -1)) 
            var             = K.reshape(var, (batch_size, n_frames, -1)) 
            logvar          = K.reshape(logvar, (batch_size, n_frames, -1)) 

            # E[log_pXz] ~= log_pXz
            log_pXz         = K.square(y_true_reshape - mean)/var
            log_pXz         = K.sum(0.5*log_pXz, axis=-1)
            
            # KL divergence between q(z|x) and p(z)
            kl_loss         = -0.5 * K.sum(1 + Z_log_var - K.square(Z_mean) - K.exp(Z_log_var), axis=-1)

            # HVRAE loss is log_pXz + kl_loss
            HVRAE_loss        = K.mean(log_pXz + kl_loss) # Do mean over batches
            return HVRAE_loss

        # Define stochastic gradient descent optimizer Adam
        adam    = optimizers.Adam(lr=0.001, beta_1=0.9, beta_2=0.999, amsgrad=False)
        # Compile the model
        self.HVRAE_mdl.compile(optimizer=adam, loss=HVRAE_loss)

        # Train the model
        self.HVRAE_mdl.fit(train_data, train_data, # Train on the normal training dataset in an unsupervised way
                epochs=20,
                batch_size=8,
                shuffle=False,
                validation_data=(test_data, test_data), # Testing on the normal tesing dataset
                callbacks=[TensorBoard(log_dir=(self.model_dir))])
        self.HVRAE_mdl.save(self.model_dir + 'HVRAE_mdl.h5')
        # plot_model(self.HVRAE_mdl, show_shapes =True, to_file=self.model_dir+'HVRAE_mdl_online.png')
        print("INFO: Training is done!")
        print("*********************************************************************")

    def HVRAE_predict(self, inferencedata):# add reltime centroid z
        K.clear_session()

        def sampling_predict(args): # Instead of sampling from Q(z|X), sample epsilon = N(0,I), z = z_mean + sqrt(var) * epsilon
            Z_mean, Z_log_var   = args
            batch_size          = K.shape(Z_mean)[0]
            n_frames            = K.int_shape(Z_mean)[1]
            n_latentdim         = K.int_shape(Z_mean)[2]
            # For reproducibility, we set the seed=37
            epsilon             = K.random_normal(shape=(batch_size, n_frames, n_latentdim), mean=0., stddev=1.0, seed=None)
            Z                   = Z_mean + K.exp(0.5*Z_log_var) * epsilon # The reparameterization trick
            return  Z

        # Load saved model
        model = load_model(self.model_dir + 'Saved_modelsHVRAE_old_mdl.h5', compile = False, custom_objects={'SamplingLayer': SamplingLayer, 'tf': tf})
        adam  = optimizers.Adam(lr=0.001, beta_1=0.9, beta_2=0.999, amsgrad=False)
        # Because we do not train the model, the loss function does not matter here.
        # Adding MSE as loss is omly for compiling the model. We can add any loss function here.
        # This is because our HVRAE loss function is customized function, we can not simply add it here.
        # We will define and call the HVRAE loss later.
        model.compile(optimizer=adam, loss=mse)
        print("INFO: Model loaded from " + self.model_dir + 'HVRAE_mdl.h5')

        get_z_mean_model    = Model(inputs=model.input, outputs=model.get_layer('qzx_mean').output)
        get_z_log_var_model = Model(inputs=model.input, outputs=model.get_layer('qzx_log_var').output)

        # Numpy version of HVRAE_loss function
        def HVRAE_loss(y_true, y_pred, Z_mean, Z_log_var):
            batch_size      = y_true.shape[0]
            n_frames        = y_true.shape[1]
            n_features      = y_true.shape[-1]

            mean            = y_pred[:, :, :, :n_features]
            logvar          = y_pred[:, :, :, n_features:]
            var             = np.exp(logvar)

            y_true_reshape  = np.reshape(y_true, (batch_size, n_frames, -1)) 
            mean            = np.reshape(mean, (batch_size, n_frames, -1)) 
            var             = np.reshape(var, (batch_size, n_frames, -1)) 
            logvar          = np.reshape(logvar, (batch_size, n_frames, -1)) 

            # E[log_pXz] ~= log_pXz
            # log_pXz       = K.square(y_true_reshape-mean)/var + logvar
            log_pXz         = np.square(y_true_reshape - mean)/var
            log_pXz         = np.sum(0.5*log_pXz, axis=-1)
            
            # KL divergence between q(z|x) and p(z)
            kl_loss         = -0.5 * np.sum(1 + Z_log_var - np.square(Z_mean) - np.exp(Z_log_var), axis=-1)

            # HVRAE loss is log_pXz + kl_loss
            HVRAE_loss        = np.mean(log_pXz + kl_loss) # Do mean over batches
            return HVRAE_loss

        print("INFO: Start to predict...")
        prediction_history  = []
        loss_history        = []
        for pattern in inferencedata:
            pattern             = np.expand_dims(pattern, axis=0)
            current_prediction  = model.predict(pattern, batch_size=1)
            predicted_z_mean    = get_z_mean_model.predict(pattern, batch_size=1)
            predicted_z_log_var = get_z_log_var_model.predict(pattern, batch_size=1)
            # Call the HVRAE_loss function
            # The HVRAE_loss function input is: 
            # Model input motion pattern, model output mean and logvar of p(X|z), mean of q(z|X), logvar of q(z|X)
            current_loss        = HVRAE_loss(pattern, current_prediction, predicted_z_mean, predicted_z_log_var)
            loss_history.append(current_loss)
        print("INFO: Prediction is done!")
        
        return loss_history

class compute_metric:
    def __init__(self):
        pass

    def detect_falls(self, loss_history, centroidZ_history, threshold):
        assert len(loss_history) == len(centroidZ_history), "ERROR: The length of loss history is different than the length of centroidZ history!"
        seq_len                 = len(loss_history)
        win_len                 = 40 # Detection window length on account of 2 seconds for 20 fps radar rate
        centroidZ_dropthres     = 0.8
        i                       = int(win_len/2)
        detected_falls_idx      = []
        # Firstly, detect the fall centers based on the centroidZ drop
        while i < (seq_len - win_len/2): 
            detection_window_middle  = i
            detection_window_lf_edge = int(detection_window_middle - win_len/2)
            detection_window_rh_edge = int(detection_window_middle + win_len/2)
            # Search the centroidZ drop
            if centroidZ_history[detection_window_lf_edge] - centroidZ_history[detection_window_rh_edge] >= centroidZ_dropthres:
                detected_falls_idx.append(int(detection_window_middle))
            i += 1

        # Secondly, if a sequence of fall happen within a window less than win_len, we combine these falls into one fall centered at the middle of this sequence
        i = 0
        processed_detected_falls_idx = []
        while i < len(detected_falls_idx):
            j = i
            while True:
                if j == len(detected_falls_idx):
                    break 
                if detected_falls_idx[j] - detected_falls_idx[i] > win_len:
                    break
                j += 1
            processed_detected_falls_idx.append(int((detected_falls_idx[i] + detected_falls_idx[j-1])/2))
            i = j

        # Thirdly, find id there is an anomaly level (or loss history) spike in the detection window
        ones_idx                    = np.argwhere(np.array(loss_history)>=threshold).flatten()
        fall_binseq                 = np.zeros(seq_len)
        fall_binseq[ones_idx]       = 1
        final_detected_falls_idx    = []
        i = 0 
        while i < len(processed_detected_falls_idx):
            detection_window_middle  = int(processed_detected_falls_idx[i])
            detection_window_lf_edge = int(detection_window_middle - win_len/2)
            detection_window_rh_edge = int(detection_window_middle + win_len/2)
            if 1 in fall_binseq[detection_window_lf_edge:detection_window_rh_edge]:
                final_detected_falls_idx.append(processed_detected_falls_idx[i])
            i += 1
        
        return final_detected_falls_idx, len(processed_detected_falls_idx)

    def find_tpfpfn(self, detected_falls_idx, gt_falls_idx):
        n_detected_falls    = len(detected_falls_idx)
        falls_tp            = []
        falls_fp            = []
        falls_fn            = list(gt_falls_idx)
        win_len             = 20
        for i in range(n_detected_falls):
            n_gt_falls      = len(falls_fn)
            j               = 0
            while j < n_gt_falls:
                # Find a gt fall index whose window covers the detected fall index, so it's true positive
                if int(falls_fn[j]-win_len/2) <= detected_falls_idx[i] <= int(falls_fn[j]+win_len/2):
                    # Remove the true positive from the gt_falls_idx list, finally only false negative remains
                    falls_fn.pop(j)  
                    falls_tp.append(i)
                    break
                j += 1
            # Dn not find a gt fall index whose window covers the detected fall index, so it's false positive
            if j == n_gt_falls:
                falls_fp.append(i)

        return falls_tp, falls_fp, falls_fn

    def cal_roc(self, loss_history, centroidZ_history, gt_falls_idx):
        n_gt_falls = len(gt_falls_idx)
        print("How many falls?", n_gt_falls)
        tpr, fpr = [], []
        for threshold in np.arange(0.0, 10.0, 0.1):
            detected_falls_idx, _           = self.detect_falls(loss_history, centroidZ_history, threshold)
            falls_tp, falls_fp, falls_fn    = self.find_tpfpfn(detected_falls_idx, gt_falls_idx)
            # Save the true positve rate for this threshold.
            tpr.append(len(falls_tp)/n_gt_falls)
            # Save the number of false positve, or missed fall detection, for this threshold
            fpr.append(len(falls_fp))
        return tpr, fpr




# Change the configuration file name
#configFileName = 'profile_reza.cfg'
#configFileName = 'ODS_6m_default-beifen.cfg'
#configFileName = 'ODS_6m_default.cfg'
#configFileName = 'ODS_6m_staticRetention.cfg'
configFileName = 'ODS_6m_staticRetention_max_acceleration_edited.cfg'
#configFileName = 'ODS_6m_staticRetention_han_1027.cfg'   #Han's enhanced config
#configFileName = 'ODS_6m_default-han_edition.cfg'
#configFileName = 'ODS_6m_staticRetention_han_edition.cfg'
csv_file_path = 'radar_data.csv'
CLIport = None
Dataport = None
byteBuffer = np.zeros(2**15,dtype = 'uint8')
byteBufferLength = 0


# ------------------------------------------------------------------

# Function to configure the serial ports and send the data from
# the configuration file to the radar
def serialConfig(configFileName):
    
    global CLIport, Dataport
    # Open the serial ports for the configuration and the data ports
    
    # Raspberry pi
    #CLIport = serial.Serial('/dev/ttyACM0', 115200)
    #Dataport = serial.Serial('/dev/ttyACM1', 921600)

    # Windows
    CLIport = serial.Serial('COM11', 115200)
    Dataport = serial.Serial('COM7', 921600)


    # Read the configuration file and send it to the board
    config = [line.rstrip('\r\n') for line in open(configFileName)]
    for i in config:
        CLIport.write((i+'\n').encode())
        print(i)
        time.sleep(0.01)

    return CLIport, Dataport

# ------------------------------------------------------------------

# Function to parse the data inside the configuration file
def parseConfigFile(configFileName):
    configParameters = {} # Initialize an empty dictionary to store the configuration parameters
    
    # Read the configuration file and send it to the board
    config = [line.rstrip('\r\n') for line in open(configFileName)]
    for i in config:
        
        # Split the line
        splitWords = i.split(" ")
        
        # Hard code the number of antennas, change if other configuration is used
        numRxAnt = 4
        numTxAnt = 3
        
        # Get the information about the profile configuration
        if "profileCfg" in splitWords[0]:
            startFreq = int(float(splitWords[2]))
            idleTime = int(splitWords[3])
            rampEndTime = float(splitWords[5])
            freqSlopeConst = float(splitWords[8])
            numAdcSamples = int(splitWords[10])
            numAdcSamplesRoundTo2 = 1;
            
            while numAdcSamples > numAdcSamplesRoundTo2:
                numAdcSamplesRoundTo2 = numAdcSamplesRoundTo2 * 2;
                
            digOutSampleRate = int(splitWords[11]);
            
        # Get the information about the frame configuration    
        elif "frameCfg" in splitWords[0]:
            
            chirpStartIdx = int(splitWords[1]);
            chirpEndIdx = int(splitWords[2]);
            numLoops = int(splitWords[3]);
            numFrames = int(splitWords[4]);
            framePeriodicity = int(splitWords[5]);

            
    # Combine the read data to obtain the configuration parameters           
    numChirpsPerFrame = (chirpEndIdx - chirpStartIdx + 1) * numLoops
    configParameters["numDopplerBins"] = numChirpsPerFrame / numTxAnt
    configParameters["numRangeBins"] = numAdcSamplesRoundTo2
    configParameters["rangeResolutionMeters"] = (3e8 * digOutSampleRate * 1e3) / (2 * freqSlopeConst * 1e12 * numAdcSamples)
    configParameters["rangeIdxToMeters"] = (3e8 * digOutSampleRate * 1e3) / (2 * freqSlopeConst * 1e12 * configParameters["numRangeBins"])
    configParameters["dopplerResolutionMps"] = 3e8 / (2 * startFreq * 1e9 * (idleTime + rampEndTime) * 1e-6 * configParameters["numDopplerBins"] * numTxAnt)
    configParameters["maxRange"] = (300 * 0.9 * digOutSampleRate)/(2 * freqSlopeConst * 1e3)
    configParameters["maxVelocity"] = 3e8 / (4 * startFreq * 1e9 * (idleTime + rampEndTime) * 1e-6 * numTxAnt)
    
    return configParameters
   
# ------------------------------------------------------------------

# Funtion to read and parse the incoming data

def readAndParseData14xx(Dataport, configParameters, flush_interval=1000):
    

    global byteBuffer, byteBufferLength

    # Constants
    OBJ_STRUCT_SIZE_BYTES = 12
    BYTE_VEC_ACC_MAX_SIZE = 2**15
    MMWDEMO_OUTPUT_MSG_COMPRESSED_POINTS = 1020
    MMWDEMO_OUTPUT_MSG_TRACKERPROC_3D_TARGET_LIST = 1010
    MMWDEMO_OUTPUT_MSG_TRACKERPROC_TARGET_INDEX = 1011
    MMWDEMO_OUTPUT_MSG_TRACKERPROC_TARGET_HEIGHT = 1012
    MMWDEMO_OUTPUT_MSG_PRESCENCE_INDICATION = 1021


    maxBufferSize = 2**15
    magicWord = [2, 1, 4, 3, 6, 5, 8, 7]

    # Initialize variables
    magicOK = 0 # Checks if magic number has been read
    dataOK = 0 # Checks if the data has been read correctly
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
            check = byteBuffer[loc:loc+8]
            if np.all(check == magicWord):
                startIdx.append(loc)

        # Check that startIdx is not empty
        if startIdx:
            # Remove the data before the first start index
            if startIdx[0] > 0 and startIdx[0] < byteBufferLength:
                byteBuffer[:byteBufferLength-startIdx[0]] = byteBuffer[startIdx[0]:byteBufferLength]
                byteBuffer[byteBufferLength-startIdx[0]:] = np.zeros(len(byteBuffer[byteBufferLength-startIdx[0]:]), dtype='uint8')   #uint8? ----Han
                byteBufferLength = byteBufferLength - startIdx[0]

            # Check that there have no errors with the byte buffer length
            if byteBufferLength < 0:
                byteBufferLength = 0

            # word array to convert 4 bytes to a 32 bit number
            word = [1, 2**8, 2**16, 2**24]

            # Read the total packet length
            totalPacketLen = np.matmul(byteBuffer[12:12+4], word)

            # Check that all the packet has been read
            if (byteBufferLength >= totalPacketLen) and (byteBufferLength != 0):
                magicOK = 1

    # If magicOK is equal to 1 then process the message
    if magicOK:
        # word array to convert 4 bytes to a 32 bit number
        word = [1, 2**8, 2**16, 2**24]

        # Initialize the pointer index
        idX = 0

        # Read the header
        magicNumber = byteBuffer[idX:idX+8]
        idX += 8
        version = format(np.matmul(byteBuffer[idX:idX+4], word), 'x')
        idX += 4
        totalPacketLen = np.matmul(byteBuffer[idX:idX+4], word)
        idX += 4
        platform = format(np.matmul(byteBuffer[idX:idX+4], word), 'x')
        idX += 4
        frameNumber = np.matmul(byteBuffer[idX:idX+4], word)
        idX += 4
        timeCpuCycles = np.matmul(byteBuffer[idX:idX+4], word)
        idX += 4
        numDetectedObj = np.matmul(byteBuffer[idX:idX+4], word)
        idX += 4
        numTLVs = np.matmul(byteBuffer[idX:idX+4], word)
        idX += 4

        subFrameNumber = np.matmul(byteBuffer[idX:idX+4], word)
        idX += 4

        # Read the TLV messages
        for tlvIdx in range(numTLVs):
            word = [1, 2**8, 2**16, 2**24]

            # Check the header of the TLV message
            original = byteBuffer[idX:idX+4]
            tlv_type = np.matmul(byteBuffer[idX:idX+4], word)
            idX += 4
            tlv_length = np.matmul(byteBuffer[idX:idX+4], word)
            idX += 4
            # print('Original:', original)
            # print('TLV type:', tlv_type)

            # Read the data depending on the TLV message
            if tlv_type == MMWDEMO_OUTPUT_MSG_COMPRESSED_POINTS:
                #print('we are in 1020')

                # idX += tlv_length
                pointCloud = []

                elevationUnit = np.frombuffer(byteBuffer[idX:idX+4], dtype=np.float32)[0]
                idX += 4
                azimuthUnit = np.frombuffer(byteBuffer[idX:idX+4], dtype=np.float32)[0]
                idX += 4
                dopplerUnit = np.frombuffer(byteBuffer[idX:idX+4], dtype=np.float32)[0]
                idX += 4
                rangeUnit = np.frombuffer(byteBuffer[idX:idX+4], dtype=np.float32)[0]
                idX += 4
                snrUnit = np.frombuffer(byteBuffer[idX:idX+4], dtype=np.float32)[0]
                idX += 4

                numPoints = (tlv_length - 20) // 8

                for _ in range(numPoints):
                    elevation = np.frombuffer(byteBuffer[idX:idX+1], dtype=np.int8)[0] * elevationUnit
                    idX += 1
                    azimuth = np.frombuffer(byteBuffer[idX:idX+1], dtype=np.int8)[0] * azimuthUnit
                    idX += 1
                    doppler = np.frombuffer(byteBuffer[idX:idX+2], dtype=np.int16)[0] * dopplerUnit
                    idX += 2
                    rangeVal = np.frombuffer(byteBuffer[idX:idX+2], dtype=np.uint16)[0] * rangeUnit
                    idX += 2
                    snr = np.frombuffer(byteBuffer[idX:idX+2], dtype=np.uint16)[0] * snrUnit
                    idX += 2

                    # Convert spherical to Cartesian coordinates
                    x = rangeVal * np.cos(elevation) * np.sin(azimuth)
                    y = rangeVal * np.cos(elevation) * np.cos(azimuth)
                    z = rangeVal * np.sin(elevation)

                    pointCloud.append([x, y, z, doppler, snr])

                detObj['pointCloud'] = pointCloud
                dataOK = 1

            elif tlv_type == MMWDEMO_OUTPUT_MSG_TRACKERPROC_3D_TARGET_LIST:  #problem
                #print('we are in 1010')

                # idX += tlv_length

                #print("tlv_length", tlv_length)

                # orig_idX = idX

                targetList = []

                # numTargets = (tlv_length - 8) // (4 + 4*13 + 4*16 + 4*2) + 1   # can be solved by + 1 at the end but don't know why
                numTargets = tlv_length // (4*12 + 4*16)

                for _ in range(numTargets):
                    tid = np.frombuffer(byteBuffer[idX:idX+4], dtype=np.uint32)[0]
                    idX += 4
                    posX = np.frombuffer(byteBuffer[idX:idX+4], dtype=np.float32)[0]
                    idX += 4
                    posY = np.frombuffer(byteBuffer[idX:idX+4], dtype=np.float32)[0]
                    idX += 4
                    posZ = np.frombuffer(byteBuffer[idX:idX+4], dtype=np.float32)[0]
                    idX += 4
                    velX = np.frombuffer(byteBuffer[idX:idX+4], dtype=np.float32)[0]
                    idX += 4
                    velY = np.frombuffer(byteBuffer[idX:idX+4], dtype=np.float32)[0]
                    idX += 4
                    velZ = np.frombuffer(byteBuffer[idX:idX+4], dtype=np.float32)[0]
                    idX += 4
                    accX = np.frombuffer(byteBuffer[idX:idX+4], dtype=np.float32)[0]
                    idX += 4
                    accY = np.frombuffer(byteBuffer[idX:idX+4], dtype=np.float32)[0]
                    idX += 4
                    accZ = np.frombuffer(byteBuffer[idX:idX+4], dtype=np.float32)[0]
                    idX += 4
                    EC = np.frombuffer(byteBuffer[idX:idX+64], dtype=np.float32).reshape(4, 4)
                    idX += 64
                    G = np.frombuffer(byteBuffer[idX:idX+4], dtype=np.float32)[0]
                    idX += 4
                    confidenceLevel = np.frombuffer(byteBuffer[idX:idX+4], dtype=np.float32)[0]
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
                #print('we are in 1011')
                #print("tlvlength:", tlv_length)
                #orig_idX = idX
                #idX += tlv_length 
                targetIndices = []

                numPoints = tlv_length

                for _ in range(numPoints):
                    targetID = np.frombuffer(byteBuffer[idX:idX+1], dtype=np.uint8)[0]
                    idX += 1
                    targetIndices.append(targetID)

                detObj['targetIndices'] = targetIndices
                #diff = idX - orig_idX
                #print("diff:",diff)

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
        if idX > 0 and byteBufferLength > idX: #why "byteBufferLength > idX"? -----Xilai
            shiftSize = totalPacketLen
            byteBuffer[:byteBufferLength - shiftSize] = byteBuffer[shiftSize:byteBufferLength]
            byteBuffer[byteBufferLength - shiftSize:] = np.zeros(len(byteBuffer[byteBufferLength - shiftSize:]), dtype='uint8')
            byteBufferLength = byteBufferLength - shiftSize

            # Check that there are no errors with the buffer length
            if byteBufferLength < 0:
                byteBufferLength = 0

    return dataOK, frameNumber, detObj



#################################################################
# thois just foir polotting

    # global byteBuffer, byteBufferLength
    
    # # Constants
    # OBJ_STRUCT_SIZE_BYTES = 12;
    # BYTE_VEC_ACC_MAX_SIZE = 2**15;
    # MMWDEMO_UART_MSG_DETECTED_POINTS = 1;
    # MMWDEMO_UART_MSG_RANGE_PROFILE   = 2;
    # maxBufferSize = 2**15;
    # magicWord = [2, 1, 4, 3, 6, 5, 8, 7]
    
    # # Initialize variables
    # magicOK = 0 # Checks if magic number has been read
    # dataOK = 0 # Checks if the data has been read correctly
    # frameNumber = 0
    # detObj = {}
    
    # readBuffer = Dataport.read(Dataport.in_waiting)
    # byteVec = np.frombuffer(readBuffer, dtype = 'uint8')
    # byteCount = len(byteVec)
    
    # # Check that the buffer is not full, and then add the data to the buffer
    # if (byteBufferLength + byteCount) < maxBufferSize:
    #     byteBuffer[byteBufferLength:byteBufferLength + byteCount] = byteVec[:byteCount]
    #     byteBufferLength = byteBufferLength + byteCount
        
    # # Check that the buffer has some data
    # if byteBufferLength > 16:
        
    #     # Check for all possible locations of the magic word
    #     possibleLocs = np.where(byteBuffer == magicWord[0])[0]

    #     # Confirm that is the beginning of the magic word and store the index in startIdx
    #     startIdx = []
    #     for loc in possibleLocs:
    #         check = byteBuffer[loc:loc+8]
    #         if np.all(check == magicWord):
    #             startIdx.append(loc)
               
    #     # Check that startIdx is not empty
    #     if startIdx:
            
    #         # Remove the data before the first start index
    #         if startIdx[0] > 0 and startIdx[0] < byteBufferLength:
    #             byteBuffer[:byteBufferLength-startIdx[0]] = byteBuffer[startIdx[0]:byteBufferLength]
    #             byteBuffer[byteBufferLength-startIdx[0]:] = np.zeros(len(byteBuffer[byteBufferLength-startIdx[0]:]),dtype = 'uint8')
    #             byteBufferLength = byteBufferLength - startIdx[0]
                
    #         # Check that there have no errors with the byte buffer length
    #         if byteBufferLength < 0:
    #             byteBufferLength = 0
                
    #         # word array to convert 4 bytes to a 32 bit number
    #         word = [1, 2**8, 2**16, 2**24]
            
    #         # Read the total packet length
    #         totalPacketLen = np.matmul(byteBuffer[12:12+4],word)
            
    #         # Check that all the packet has been read
    #         if (byteBufferLength >= totalPacketLen) and (byteBufferLength != 0):
    #             magicOK = 1
    
    # # If magicOK is equal to 1 then process the message
    # if magicOK:
    #     # word array to convert 4 bytes to a 32 bit number
    #     word = [1, 2**8, 2**16, 2**24]
        
    #     # Initialize the pointer index
    #     idX = 0
        
    #     # Read the header
    #     magicNumber = byteBuffer[idX:idX+8]
    #     idX += 8
    #     version = format(np.matmul(byteBuffer[idX:idX+4],word),'x')
    #     idX += 4
    #     totalPacketLen = np.matmul(byteBuffer[idX:idX+4],word)
    #     idX += 4
    #     platform = format(np.matmul(byteBuffer[idX:idX+4],word),'x')
    #     idX += 4
    #     frameNumber = np.matmul(byteBuffer[idX:idX+4],word)
    #     idX += 4
    #     timeCpuCycles = np.matmul(byteBuffer[idX:idX+4],word)
    #     idX += 4
    #     numDetectedObj = np.matmul(byteBuffer[idX:idX+4],word)
    #     idX += 4
    #     numTLVs = np.matmul(byteBuffer[idX:idX+4],word)
    #     print('NumTLVs:', numTLVs)
    #     idX += 4

       

    #     # UNCOMMENT IN CASE OF SDK 2
    #     subFrameNumber = np.matmul(byteBuffer[idX:idX+4],word)
    #     print('Subframwe:', subFrameNumber)

    #     idX += 4
        
    #     # Read the TLV messages
    #     for tlvIdx in range(numTLVs):
            
    #         # print('range tlv = ',  tlvIdx )
    #         # word array to convert 4 bytes to a 32 bit number
    #         word = [1, 2**8, 2**16, 2**24]

    #         # Check the header of the TLV message
    #         tlv_type = np.matmul(byteBuffer[idX:idX+4],word)
            
    #         idX += 4
    #         tlv_length = np.matmul(byteBuffer[idX:idX+4],word)
    #         idX += 4
            
    #         # Read the data depending on the TLV message
    #         if tlv_type == 1020:
    #             # Parse the 3D Spherical Compressed Point Cloud data
    #             pointCloud = []

    #             elevationUnit = np.frombuffer(byteBuffer[idX:idX+4], dtype=np.float32)[0]
    #             idX += 4
    #             azimuthUnit = np.frombuffer(byteBuffer[idX:idX+4], dtype=np.float32)[0]
    #             idX += 4
    #             dopplerUnit = np.frombuffer(byteBuffer[idX:idX+4], dtype=np.float32)[0]
    #             idX += 4
    #             rangeUnit = np.frombuffer(byteBuffer[idX:idX+4], dtype=np.float32)[0]
    #             idX += 4
    #             snrUnit = np.frombuffer(byteBuffer[idX:idX+4], dtype=np.float32)[0]
    #             idX += 4

    #             numPoints = (tlv_length - 20) // 8

    #             for _ in range(numPoints):
    #                 elevation = np.frombuffer(byteBuffer[idX:idX+1], dtype=np.int8)[0] * elevationUnit
    #                 idX += 1
    #                 azimuth = np.frombuffer(byteBuffer[idX:idX+1], dtype=np.int8)[0] * azimuthUnit
    #                 idX += 1
    #                 doppler = np.frombuffer(byteBuffer[idX:idX+2], dtype=np.int16)[0] * dopplerUnit
    #                 idX += 2
    #                 rangeVal = np.frombuffer(byteBuffer[idX:idX+2], dtype=np.uint16)[0] * rangeUnit
    #                 idX += 2
    #                 snr = np.frombuffer(byteBuffer[idX:idX+2], dtype=np.uint16)[0] * snrUnit
    #                 idX += 2

    #                 # Convert spherical to Cartesian coordinates
    #                 x = rangeVal * np.cos(elevation) * np.sin(azimuth)
    #                 y = rangeVal * np.cos(elevation) * np.cos(azimuth)
    #                 z = rangeVal * np.sin(elevation)

    #                 pointCloud.append([x, y, z, doppler, snr])
    #                 #print(pointCloud)

    #             detObj['pointCloud'] = pointCloud
    #             dataOK = 1
    #         elif tlv_type == 1010:
    #             print(' IT EXISTS')
                


    #     # Remove already processed data
    #     if idX > 0 and byteBufferLength > idX:
    #         shiftSize = totalPacketLen
                
    #         byteBuffer[:byteBufferLength - shiftSize] = byteBuffer[shiftSize:byteBufferLength]
    #         byteBuffer[byteBufferLength - shiftSize:] = np.zeros(len(byteBuffer[byteBufferLength - shiftSize:]), dtype='uint8')
    #         byteBufferLength = byteBufferLength - shiftSize
            
    #         # Check that there are no errors with the buffer length
    #         if byteBufferLength < 0:
    #             byteBufferLength = 0

    # return dataOK, frameNumber, detObj
  


############################### MAIN ####################################

import tensorflow as tf
import sys
import numpy as np
import pandas as pd
import pyqtgraph.opengl as gl
from PyQt5 import QtWidgets, QtCore
import tensorflow as tf
import serial
from pyqtgraph.Qt import QtGui
import pyqtgraph as pg


recent_predictions = []

current_frame_data = None
current_frame_number = None
fall_detected = False
# Initialize global variables for frame data handling
current_frame_data = pd.DataFrame()
current_frame_number = -1

data_processor = data_preproc()

import sys
import os
import pandas as pd
import numpy as np
from PyQt5 import QtWidgets, QtCore, QtGui
import pyqtgraph.opengl as gl
import random

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
        self.create_occupancy_grid(cube_width=5, cube_height=3, cube_depth=5, grid_width=10, grid_height=10, spacing=0.5, cube_y_translation=0)

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
        self.fall_indicator.setStyleSheet("QLabel { background-color: green; border: 1px solid black; font-size: 18pt; }")
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

    def create_occupancy_grid(self, cube_width, cube_height, cube_depth, grid_width, grid_height, spacing, cube_y_translation):
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
            lines.append([np.array([grid_start_x, y, z_position], dtype=np.float32), np.array([grid_start_x + grid_width, y, z_position], dtype=np.float32)])
        for x in np.arange(grid_start_x, grid_start_x + grid_width + spacing, spacing):
            lines.append([np.array([x, grid_start_y, z_position], dtype=np.float32), np.array([x, grid_start_y + grid_height, z_position], dtype=np.float32)])

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
        edges = np.array([[0, 1], [1, 2], [2, 3], [3, 0], [4, 5], [5, 6], [6, 7], [7, 4], [0, 4], [1, 5], [2, 6], [3, 7]])
        cube_lines = [gl.GLLinePlotItem(pos=np.array([verts[edge[0]], verts[edge[1]]], dtype=np.float32), color=(1, 0, 0, 1), width=2, antialias=True) for edge in edges]
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

# ------------------- Update Function --------------------------------------------
import os
import pandas as pd
import warnings

warnings.simplefilter(action='ignore', category=FutureWarning)
# Initialize global variables to keep track of previous data and cumulative storage

import torch
import random

# original
# def num_to_class(num):
#     if num == 0:
#         return "Falling+LayFloor"
#     elif num == 1:
#         return "LayBed"
#     elif num == 2:
#         return "Sit+SitBed"
#     elif num == 3:
#         return "Stand+Walking"
#     else:
#         return "non sense"

#5 cls with transition removed:
# def num_to_class(num):
#     if num == 0:
#         return "Falling"
#     elif num == 1:
#         return "LayBed+LayFloor"
#     elif num == 2:
#         return "Sit+SitBed"
#     elif num == 3:
#         return "Stand+Walking"
#     else:
#         return "non sense"


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


alarm_trigger = False
window = 20
current_window_idx = 0
all_data_frame = []
fall_df = pd.DataFrame(columns = ['detected_falls_idx'])

model = torch.load(r'./epoch155.pt', map_location=torch.device('cpu'))

model.eval()
device=torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model = model.to(device)


previous_pc = "empty"
previous_df = pd.DataFrame()
csv_file_path = 'output/radar_data_24.10.21.night.csv'
average_zs = []

model_output_window = []
current_output_index = 0
def append_to_csv(output, file_path):
    timestamp = time.strftime("%Y-%m-%d %H:%M:%S")  # Get current time in readable format
    with open(file_path, mode='a', newline='') as file:
        writer = csv.writer(file)
        writer.writerow([timestamp, output])
        
output_csv_file_path = "output/radar_output_log.csv"
# Function to save DataFrame to CSV
def save_to_csv(df, file_path):
    if not df.empty:
        df.to_csv(file_path, mode='a', index=False, header=not os.path.isfile(file_path))
def update():
    global s, recent_predictions, radar_gui, alarm_trigger, window, current_window_idx, all_data_frame, fall_df
    global previous_pc, previous_df, average_zs, model_output_window, current_output_index

    dataOk, frameNumber, detObj = readAndParseData14xx(Dataport, configParameters)

    if radar_gui.is_recording:
        if dataOk:
            pointCloud = np.array(detObj['pointCloud'])  # (N, 5) array: X, Y, Z, velocity, snr
            targetIndices = detObj.get('targetIndices', [])  # List of target indices
            
            # Create a DataFrame for the point cloud
            df = pd.DataFrame({
                'Frame Number': frameNumber,
                'X': pointCloud[:, 0],
                'Y': pointCloud[:, 1],
                'Z': pointCloud[:, 2],
                'velocity': pointCloud[:, 3],
                'snr': pointCloud[:, 4],
            })
            
            # Stack the X, Y, Z columns to form the point cloud shape (N, 3)
            pointCloud1 = np.vstack((df['X'], df['Y'], df['Z'], df['velocity'])).T

            # Check if previous_df is empty (first frame case)
            if previous_df.empty:
                previous_df = df

            # Check if previous_pc is empty or has no points (first frame case)
            # if  previous_pc == "empty" or previous_pc.shape[0] == 0:
            #     previous_pc = pointCloud1
            #     return
            if isinstance(previous_pc, str) and previous_pc == "empty" or previous_pc.size == 0:
                previous_pc = pointCloud1
                return

            # Store the current point cloud temporarily and update `previous_pc`
            current_pc = pointCloud1
            points = previous_pc  # Use the previous frame's points to match target IDs
            previous_pc = current_pc  # Update previous_pc with the current point cloud for the next iteration

            # Handle cases where target indices do not match point cloud length
            if len(targetIndices) != points.shape[0]:
                # print(f"Skipping frame due to mismatch: {len(targetIndices)} target IDs vs {points.shape[0]} points.")
                previous_df = df
                return

            # Save the previous frame's DataFrame with the associated target IDs
            if len(targetIndices) == len(previous_df):
                previous_df['target_id'] = targetIndices
            else:
                # print(f"Skipping target_id assignment: mismatch in lengths.")
                previous_df['target_id'] = [-1] * len(previous_df)  # Assign a default value (-1) for missing target IDs
            
            save_to_csv(previous_df, csv_file_path)  # Save previous_df to CSV
            previous_df = df  # Update for the next frame

            # Create a mask to filter out invalid target IDs (e.g., 253, 254, 255)
            mask = np.array([(0 <= x <= 255) for x in targetIndices])  # True for valid IDs

            # Apply the mask to filter out invalid points
            valid_points = points[mask]  # Filtered point cloud (N, 3)
            valid_target_ids = np.array(targetIndices)[mask]  # Corresponding valid target IDs (N,)

            # Prepare the points for visualization by adding the target ID as a fourth dimension
            points_with_ids = np.hstack((valid_points, valid_target_ids[:, np.newaxis]))

            z_values = valid_points[:, 2]
            z_mean = np.mean(z_values)
            average_zs.append(z_mean)
            valid_points = valid_points.tolist()

            if len(valid_points) == 0:
                return

            filled_frame = fill_frame(valid_points, target_length=150)  # Now filled_frame is a raw Python list of points in ONE frame

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
                    radar_gui.update_fall_indicator(afterward_output)

                    
                    # Update the GUI to reflect the activity detected
                    
                    
                    # Now append it to the CSV file along with the timestamp
                    append_to_csv(afterward_output, output_csv_file_path)
                    
                    current_output_index = 5
                    model_output_window = model_output_window[5:]

                current_window_idx = 15
                all_data_frame = all_data_frame[5:]

            radar_gui.update_scatter_plot_with_colors(points_with_ids, size=5)
            QtGui.QGuiApplication.processEvents()
    else:
        pass

def close_ports():
    if CLIport and CLIport.is_open:
        CLIport.write('sensorStop\n'.encode())
        time.sleep(0.1)  # 确保命令发送完成
        CLIport.reset_input_buffer()
        CLIport.reset_output_buffer()
        CLIport.close()
    if Dataport and Dataport.is_open:
        Dataport.close()

atexit.register(close_ports)

##############################################
# Set up the serial connection and radar configuration parameters
CLIport, Dataport = serialConfig(configFileName)
configParameters = parseConfigFile(configFileName)

# Initialize the Qt Application and RadarGUI
app = QtWidgets.QApplication([])
radar_gui = RadarGUI()
radar_gui.setWindowTitle('3D Radar Scatter Plot')
radar_gui.show()

# Connect the update function to a timer for periodic updates
timer = QtCore.QTimer()
timer.timeout.connect(update)
timer.start(33)  # Update every 33 milliseconds

# Start the Qt event loop
sys.exit(app.exec_())


########################################################################
########################################################################
#######################################################################
#######################################################################
########################################################################
