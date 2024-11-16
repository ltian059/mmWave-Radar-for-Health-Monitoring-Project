import serial
import configparser

# Initialize the ConfigParser
config = configparser.ConfigParser()
# Read the config.ini file
config.read('config.ini')  # Ensure 'config.ini' is in the same directory, or provide the full path
# Access values from the DEFAULT section
model_path = config['DEFAULT'].get('model_path', './epoch155.pt')
output_csv_file_path = config['DEFAULT'].get('output_csv_file_path', 'output/radar_output_log.csv')
csv_file_path = config['DEFAULT'].get('csv_file_path', 'output/radar_data.csv')
configFileName = config['DEFAULT'].get('configFileName', 'ODS_6m_staticRetention_max_acceleration_edited.cfg')
csv_file_path_timestamp = None


# Print out the values (optional, for debugging)
print("Model Path:", model_path)
print("Output CSV File Path:", output_csv_file_path)
print("CSV File Path:", csv_file_path)
print("Config File Name:", configFileName)


# Function to configure the serial ports and send the data from
def serialConfig(configFileName):
    global CLIport, Dataport
    # Open the serial ports for the configuration and the data ports
    # Raspberry pi
    CLIport = serial.Serial('/dev/ttyACM0', 115200)
    Dataport = serial.Serial('/dev/ttyACM1', 921600)
    # Windows
    # CLIport = serial.Serial('COM11', 115200)
    # Dataport = serial.Serial('COM7', 921600)

    # Read the configuration file and send it to the board
    # config = [line.rstrip('\r\n') for line in open(configFileName)]
    # for i in config:
    #     CLIport.write((i + '\n').encode())
    #     print(i)
    #     time.sleep(0.1)
    return CLIport, Dataport

# ------------------------------------------------------------------

# Function to parse the data inside the configuration file
def parseConfigFile(configFileName):
    configParameters = {}  # Initialize an empty dictionary to store the configuration parameters

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
    configParameters["rangeResolutionMeters"] = (3e8 * digOutSampleRate * 1e3) / (
                2 * freqSlopeConst * 1e12 * numAdcSamples)
    configParameters["rangeIdxToMeters"] = (3e8 * digOutSampleRate * 1e3) / (
                2 * freqSlopeConst * 1e12 * configParameters["numRangeBins"])
    configParameters["dopplerResolutionMps"] = 3e8 / (
                2 * startFreq * 1e9 * (idleTime + rampEndTime) * 1e-6 * configParameters["numDopplerBins"] * numTxAnt)
    configParameters["maxRange"] = (300 * 0.9 * digOutSampleRate) / (2 * freqSlopeConst * 1e3)
    configParameters["maxVelocity"] = 3e8 / (4 * startFreq * 1e9 * (idleTime + rampEndTime) * 1e-6 * numTxAnt)

    return configParameters


