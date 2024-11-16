import atexit
#import matplotlib.pyplot as plt
import time
import numpy as np
from keras import backend as K
from keras import optimizers
from keras.layers import Input, Dense, Flatten, Lambda, Concatenate, Reshape, \
    TimeDistributed, RepeatVector, SimpleRNN
from keras.models import Model, load_model
from keras.callbacks import TensorBoard
from keras.losses import MeanSquaredError

import utils

mse = MeanSquaredError()
# from keras.losses import mse
#from scipy.signal import find_peaks
#from sklearn.metrics import confusion_matrix
from tensorflow.keras import layers

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
CLIport = None
Dataport = None



# ------------------------------------------------------------------


# ------------------------------------------------------------------

# Funtion to read and parse the incoming data




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
import pandas as pd
import tensorflow as tf

recent_predictions = []

current_frame_data = None
current_frame_number = None
fall_detected = False
# Initialize global variables for frame data handling
current_frame_data = pd.DataFrame()
current_frame_number = -1

data_processor = data_preproc()

import sys
import numpy as np

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


import Configuration
from utils import close_ports
from RadarGUI import RadarGUI
import tkinter as tk

def on_exit():
    close_ports(CLIport, Dataport)

sys.path.append('./lib')
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if device.type == "cuda":
    print("Using GPU for inference", flush=True)
else:
    print("Using CPU for inference", flush=True)

model = torch.load(Configuration.model_path, map_location=device)
model = model.to(device)
model.eval()

atexit.register(on_exit)


###########################Start GUI################################
root = tk.Tk()
gui = RadarGUI(root ,model, device)
root.mainloop()

import matplotlib.pyplot as plt

def plot_frame_sequence(frame_numbers):
    plt.figure(figsize=(10, 4))
    plt.plot(frame_numbers, marker='o')
    plt.title('Processed Frame Numbers Over Time')
    plt.xlabel('Sequence Index')
    plt.ylabel('Frame Number')
    plt.grid(True)
    plt.show()

# Call this function at appropriate intervals or at the end of data collection
plot_frame_sequence(utils.processed_frame_numbers)

