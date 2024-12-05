import serial
from Configuration import serialConfig_win
import Configuration, time, sys

configFileName = Configuration.configFileName
CLIport, Dataport = serialConfig_win(configFileName)

def close_ports():
    if CLIport and CLIport.is_open:
        CLIport.write('sensorStop\n'.encode())
        time.sleep(0.1)  # 确保命令发送完成
        CLIport.reset_input_buffer()
        CLIport.reset_output_buffer()
        CLIport.close()
    if Dataport and Dataport.is_open:
        Dataport.close()


import signal

def signal_handler(sig, frame):
    close_ports()
    sys.exit(0)

signal.signal(signal.SIGINT, signal_handler)

while(True):
    pass