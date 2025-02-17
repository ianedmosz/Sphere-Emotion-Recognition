# Imports for P300
import multiprocessing
import simpleaudio as sa
import simpleaudio.functionchecks as fc
import time
from multiprocessing import Process, Value
import numpy as np
import pandas as pd
from datetime import datetime
from colorama import Fore, Style
import scipy.stats as stats
import subprocess, json
from scipy.io import wavfile

# Imports for Machine Learning
from pickle import load
from sklearn.preprocessing import MinMaxScaler
from math import floor
from sklearn.impute import KNNImputer

# Imports for OpenBCI
import os
import time
from datetime import datetime
import pandas as pd
import matplotlib.pyplot as plt
from brainflow.board_shim import BoardShim, BrainFlowInputParams, BoardIds, LogLevels
from brainflow.data_filter import DataFilter, WindowFunctions, DetrendOperations ##WindowOperations?13
import csv
from brainflow import BoardShim, BrainFlowInputParams, LogLevels, BoardIds
import pickle
# Imports from Empatica
import socket

import time
import pylsl
import numpy as np
import pandas as pd
from datetime import datetime
import csv
import matplotlib.pyplot as plt 
import matplotlib.animation as animation
from scipy.fft import rfft, rfftfreq
import json
import sys 
from math import log

# Imports for OSC
import argparse
import random
import time
from pythonosc import udp_client
from statistics import mean, median
from math import sqrt
from pathlib import Path
from serial import Serial



import pandas as pd
from brainflow import BoardShim, BrainFlowInputParams, LogLevels, BoardIds
import time
from scipy import signal
import numpy as np
from scipy.signal import welch, butter, lfilter
import pickle
import pyeeg as pe                          
from statistics import mean
import matplotlib.pyplot as plt
from PIL import Image
from pythonosc import udp_client
import random   #/usr/lib/jvm/default/bin:/
import warnings
import serial
import math
from sklearn.exceptions import DataConversionWarning
import datetime
import os
from colorama import Fore, Style
import json
from serial import Serial
from pathlib import Path


## CAMBIAR POR CADA ESCENARIO - DURACIÓN TOTAL DEL CODIGO
simulation_time = 210 ## 


# Un-comment the next line of code if sender and connector files are in another directory. And change the "modules" name 
# to your directory.
#sys.path.append("./modules")
from sender import Sender

# # CODE FOR P300 TEST # #
# # You can play a sound with your speakers with the next line, uncomment if needed to run a diagnosis # #
#fc.LeftRightCheck.run()

# # Create objects that store the beep sounds using simpleaudio # #
# This sounds can include a method called .wait_done() that will literally pause everything
# until the sound has finished.

# WARNING: The sound files must be on the same directory to have the relative path, however if they are
# on different directories, you must add the ENTIRE path, example: (r"C:\user\directory\TransitionBeep.wav")
# Do NOT forget to include the "r" before the double quotes, else it will cause an error.


seconds = Value("i", 0)
counts = Value("i", 0)

def timer(second, count, timestamps):
    
    global simulation_time
    
    # First we initialize a variable that will contain the moment the timer began and 
    # we store this in the timestamps list that will be stored in a CSV.    
    time_start = time.time()
    timestamps.append(time_start)
    while True:
        # The .get_lock() function is necessary since it ensures they are 
        # sincronized between both functions, since they both access to the same 
        # variables
        with second.get_lock(), count.get_lock():
            
            # We now calculate the time elappsed between start and now. 
            # (should be approx. 1 second)
            second.value = int(time.time() - time_start)
            count.value = 0
            if(second.value == simulation_time):
                return
            print(second.value, end="\r")
        # Once we stored all the info and make the calculations, we sleep the script for
        # one second. This is the magic of the script, it executes every  ~1 second.
        time.sleep(1) #0.996

# # CODE FOR EMPATICA# #
# Function to generate CSVs and post them to the ALAS server.
def jsonPost(csvFilePath, value):
    TemporalArray = []

    with open(csvFilePath) as Document:
        data = csv.reader(Document)
        for row in data:
            #print(row[0])
            #print(row[-1])
            TemporalArray.append({ 'datetime': row[0] , value: row[-1] })

    # We post the information with the next instruction.
    #r = requests.post(url, data=json.dumps(file), headers=headers)

    s = Sender()
    s.post_empatica_record(TemporalArray)
    # print(file)
    # #print(r.text)
    # print("-----------\n\n")
    # print(TemporalArray)

def csv2JSON(csvFilePath, jsonFilePath):
    jsonArray = []

    #read csv file
    with open(csvFilePath, encoding='utf-8') as csvf: 
        #load csv file data using csv library's dictionary reader
        csvReader = csv.DictReader(csvf) 

        #convert each csv row into python dict
        for row in csvReader: 
            #add this python dict to json array
            jsonArray.append(row)

    #convert python jsonArray to JSON String and write to file
    with open(jsonFilePath, 'w', encoding='utf-8') as jsonf: 
        jsonString = json.dumps(jsonArray, indent=4)
        jsonf.write(jsonString)

def empatica(second, folder, client, client2):
    global BVP_array, Acc_array, GSR_array, Temp_array, IBI_array
    global BVP_tuple, ACC_tuple, GSR_tuple, Temp_tuple, IBI_tuple
    global Temporal_BVP_array, Temporal_GSR_array, Temporal_Temp_array, Temporal_IBI_array, BVP_Graph_value
    global counter
    global x_BVP_val, x_GSR_val, x_Temp_val, x_IBI_val
    global y_BVP_val, y_GSR_val, y_Temp_val, y_IBI_val
    global last_values
    last_values = dict(zip(['pressure', 'electricity', 'temperature', 'heartRate' ], [0]*4))
    #global l_bvp

    # VARIABLES USED TO STORE & GRAPH DATA
    BVP_array, Acc_array, GSR_array, Temp_array, IBI_array = [], [], [], [], []
    BVP_tuple, ACC_tuple, GSR_tuple, Temp_tuple, IBI_tuple = (), (), (), (), ()
    Temporal_BVP_array, Temporal_GSR_array, Temporal_Temp_array, Temporal_IBI_array = [], [], [], []
    #l_bvp = []

    BVP_Graph_value = None
    counter = 0 # Used to pop values from arrays to perform a "moving" graph.
    x_BVP_val, x_GSR_val, x_Temp_val, x_IBI_val = [], [], [], []
    y_BVP_val, y_GSR_val, y_Temp_val, y_IBI_val = [], [], [], []
    
    
    # CSV FILES FOR TIME-INDEPENDENT STORAGE
    with open("{}/Raw/fileBVP.csv".format(folder), 'w', newline = '') as document:
        writer = csv.writer(document)
        writer.writerow(['Datetime', 'valueBVP'])
        
    # with open("{}/Raw/fileACC.csv".format(folder), 'w', newline = '') as document:
    #     writer = csv.writer(document)
    #     writer.writerow(['Datetime', 'valueACC'])  

    with open("{}/Raw/fileEDA.csv".format(folder), 'w', newline = '') as document:
        writer = csv.writer(document)
        writer.writerow(['Datetime', 'valueEDA'])      

    with open("{}/Raw/fileTemp.csv".format(folder), 'w', newline = '') as document:
        writer = csv.writer(document)
        writer.writerow(['Datetime', 'valueTemp'])     

    with open("{}/Raw/fileIBI.csv".format(folder), 'w', newline = '') as document:
        writer = csv.writer(document)
        writer.writerow(['Datetime', 'valueIBI'])

    # SELECT DATA TO STREAM
    acc = True      # 3-axis acceleration
    bvp = True      # Blood Volume Pulse
    gsr = True      # Galvanic Skin Response (Electrodermal Activity)
    tmp = True      # Temperature
    ibi = True

    serverAddress = '127.0.0.1'  #'FW 2.1.0' #'127.0.0.1'
    serverPort = 28000 #28000 #4911
    bufferSize = 4096
    # The wristband with SN A027D2 worked here with deviceID 8839CD
    deviceID = '8839CD'  #'8839CD' #'1451CD' # 'A02088' #'A01FC2' #'de6f5a'

    escalado_11 = lambda x : (x-0.5)*2
    escalado_minmax = lambda x, min, max : (x-min)/(max-min)
    def connect():
        global s
        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        s.settimeout(3)

        print("Connecting to server")
        s.connect((serverAddress, serverPort))
        print("Connected to server\n")

        print("Devices available:")
        s.send("device_list\r\n".encode())
        response = s.recv(bufferSize)
        print(response.decode("utf-8"))

        print("Connecting to device")
        s.send(("device_connect " + deviceID + "\r\n").encode())
        response = s.recv(bufferSize)
        print(response.decode("utf-8"))

        print("Pausing data receiving")
        s.send("pause ON\r\n".encode())
        response = s.recv(bufferSize)
        print(response.decode("utf-8"))
        
    connect()

    time.sleep(1)

    def suscribe_to_data():
        if acc:
            print("Suscribing to ACC")
            s.send(("device_subscribe " + 'acc' + " ON\r\n").encode())
            response = s.recv(bufferSize)
            print(response.decode("utf-8"))
        if bvp:
            print("Suscribing to BVP")
            s.send(("device_subscribe " + 'bvp' + " ON\r\n").encode())
            response = s.recv(bufferSize)
            print(response.decode("utf-8"))
        if gsr:
            print("Suscribing to GSR")
            s.send(("device_subscribe " + 'gsr' + " ON\r\n").encode())
            response = s.recv(bufferSize)
            print(response.decode("utf-8"))
        if tmp:
            print("Suscribing to Temp")
            s.send(("device_subscribe " + 'tmp' + " ON\r\n").encode())
            response = s.recv(bufferSize)
            print(response.decode("utf-8"))
        if ibi:
            print("Suscribing to Ibi")
            s.send(("device_subscribe " + 'ibi' + " ON\r\n").encode())
            response = s.recv(bufferSize)
            print(response.decode("utf-8"))

        print("Resuming data receiving")
        s.send("pause OFF\r\n".encode())
        response = s.recv(bufferSize)
        print(response.decode("utf-8"))
    suscribe_to_data()

    def prepare_LSL_streaming():
        print("Starting LSL streaming")
        if acc:
            infoACC = pylsl.StreamInfo('acc','ACC',3,32,'int32','ACC-empatica_e4')
            global outletACC
            outletACC = pylsl.StreamOutlet(infoACC)
        if bvp:
            infoBVP = pylsl.StreamInfo('bvp','BVP',1,64,'float32','BVP-empatica_e4')
            global outletBVP
            outletBVP = pylsl.StreamOutlet(infoBVP)
        if gsr:
            infoGSR = pylsl.StreamInfo('gsr','GSR',1,4,'float32','GSR-empatica_e4')
            global outletGSR
            outletGSR = pylsl.StreamOutlet(infoGSR)
        if tmp:
            infoTemp = pylsl.StreamInfo('tmp','Temp',1,4,'float32','Temp-empatica_e4')
            global outletTemp
            outletTemp = pylsl.StreamOutlet(infoTemp)
        if ibi:
            infoIbi = pylsl.StreamInfo('ibi','Ibi',1,2,'float32','IBI-empatica_e4')
            global outletIbi
            outletIbi = pylsl.StreamOutlet(infoIbi)
    prepare_LSL_streaming()

    time.sleep(1)

    def reconnect():
        print("Reconnecting...")
        connect()
        suscribe_to_data()
        stream()

    def stream():
        global last_values
        global simulation_time
        try:
            print("Streaming...")
            try:
                with second.get_lock():
                    # When the seconds reach 312, we exit the functions.
                    if(second.value == simulation_time):
                        plt.close()
                        return
                response = s.recv(bufferSize).decode("utf-8")
                # print(response)
                if "connection lost to device" in response:
                    print(response.decode("utf-8"))
                    reconnect()
                samples = response.split("\n") # Variable "samples" contains all the information collected from the wristband.
                # print(samples)
                # We need to clean every temporal array before entering the for loop.
                global Temporal_BVP_array
                global Temporal_GSR_array
                global Temporal_Temp_array
                global Temporal_IBI_array
                global flag_Temp # We only want one value of the Temperature to reduce the final file size.
                global l_bvp
                l_bvp, l_gsr, l_tmp, l_ibi = [], [], [], []
                flag_Temp = 0
                for i in range(len(samples)-1):
                    try:
                        stream_type = samples[i].split()[0]
                    except:
                        continue
                    # print(samples)
                    # if (stream_type == "E4_Acc"):
                    #     global Acc_array
                    #     global ACC_tuple
                    #     timestamp = float(samples[i].split()[1].replace(',','.'))
                    #     data = [int(samples[i].split()[2].replace(',','.')), int(samples[i].split()[3].replace(',','.')), int(samples[i].split()[4].replace(',','.'))]
                    #     outletACC.push_sample(data, timestamp=timestamp)
                    #     timestamp = datetime.fromtimestamp(timestamp)
                    #     print(data) # Added in 02/12/20 to show values
                    #     ACC_tuple = (timestamp.strftime('%Y-%m-%d %H:%M:%S.%f'), data)
                    #     Acc_array.append(ACC_tuple)
                    #     Acc_df = pd.DataFrame(ACC_tuple)
                    #     Acc_df.to_csv("{}/Raw/fileACC.csv".format(folder), mode='a', index=False, header=False)
                        
                    if stream_type == "E4_Bvp":
                        global BVP_tuple
                        global BVP_array
                        timestamp = float(samples[i].split()[1].replace(',','.'))
                        data = float(samples[i].split()[2].replace(',','.'))
                        
                        l_bvp.append(data)
                        
                        outletBVP.push_sample([data], timestamp=timestamp)
                        timestamp = datetime.fromtimestamp(timestamp)
                        # print(data)
                        Temporal_BVP_array.append(data)
                        BVP_tuple = (timestamp.strftime('%Y-%m-%d %H:%M:%S.%f'), data)
                        BVP_array.append(BVP_tuple)
                        BVP_df = pd.DataFrame(BVP_tuple)
                        BVP_df.to_csv("{}/Raw/fileBVP.csv".format(folder), mode='a', index=False, header=False)
                        
                    if stream_type == "E4_Gsr":
                        global GSR_array
                        global GSR_tuple
                        timestamp = float(samples[i].split()[1].replace(',','.'))
                        data = float(samples[i].split()[2].replace(',','.'))

                        if type(data) == float:
                            l_gsr.append(data)

                        outletGSR.push_sample([data], timestamp=timestamp)
                        timestamp = datetime.fromtimestamp(timestamp)
                        # print(data)
                        Temporal_GSR_array.append(data)
                        GSR_tuple = (timestamp.strftime('%Y-%m-%d %H:%M:%S.%f'), data)
                        GSR_array.append(GSR_tuple)
                        EDA_df = pd.DataFrame(GSR_tuple)
                        EDA_df.to_csv("{}/Raw/fileEDA.csv".format(folder), mode='a', index=False, header=False)
                        
                    if stream_type == "E4_Temperature":
                        global Temp_array
                        global Temp_tuple
                        timestamp = float(samples[i].split()[1].replace(',','.'))
                        data = float(samples[i].split()[2].replace(',','.'))
                        
                        if type(data) == float:
                            l_tmp.append(data)

                        outletTemp.push_sample([data], timestamp=timestamp)
                        timestamp = datetime.fromtimestamp(timestamp)
                        # print(data)
                        Temporal_Temp_array.append(data)
                        
                        if flag_Temp == 0:
                            Temp_tuple = (timestamp.strftime('%Y-%m-%d %H:%M:%S.%f'), Temporal_Temp_array[0])
                            Temp_array.append(Temp_tuple)
                            Temp_df = pd.DataFrame(Temp_tuple)
                            Temp_df.to_csv("{}/Raw/fileTemp.csv".format(folder), mode='a', index=False, header=False)
                            flag_Temp = 1
                            
                    if stream_type == "E4_Ibi":
                        global IBI_array
                        global IBI_tuple
                        timestamp = float(samples[i].split()[1].replace(',','.'))
                        data = float(samples[i].split()[2].replace(',','.'))

                        if type(data) == float:
                            l_ibi.append(data)

                        data = 60/data
                        
                        outletIbi.push_sample([data], timestamp=timestamp)
                        timestamp = datetime.fromtimestamp(timestamp)
                        # print(data)
                        Temporal_IBI_array.append(data)
                        IBI_tuple = (timestamp.strftime('%Y-%m-%d %H:%M:%S.%f'), data)
                        IBI_array.append(IBI_tuple)
                        IBI_df = pd.DataFrame(IBI_tuple)
                        IBI_df.to_csv("{}/Raw/fileIBI.csv".format(folder), mode='a', index=False, header=False)
                            
                         
                    with open("{}/Raw/fileBVP.csv".format(folder), 'w', newline = '') as document:
                        writer = csv.writer(document)
                        writer.writerow(['Datetime', 'valueBVP'])
                        writer.writerows(BVP_array)

                    """ with open("{}/Raw/fileACC.csv".format(folder), 'w', newline = '') as document:
                        writer = csv.writer(document)
                        writer.writerow(['Datetime', 'valueACC'])
                        writer.writerows(Acc_array) """   

                    with open("{}/Raw/fileEDA.csv".format(folder), 'w', newline = '') as document:
                        writer = csv.writer(document)
                        writer.writerow(['Datetime', 'valueEDA'])
                        writer.writerows(GSR_array)        

                    with open("{}/Raw/fileTemp.csv".format(folder), 'w', newline = '') as document:
                        writer = csv.writer(document)
                        writer.writerow(['Datetime', 'valueTemp'])
                        writer.writerows(Temp_array)        

                    with open("{}/Raw/fileIBI.csv".format(folder), 'w', newline = '') as document:
                        writer = csv.writer(document)
                        writer.writerow(['Datetime', 'valueIBI'])
                        writer.writerows(IBI_array)   

                def process_value(l, biometric):
                    # [BVP] OSC: -100 < x < 100, fs = 64 Hz
                    # # send_data = data if abs(data) < 100 else 100
                    # # send_data = median(l_bvp)/100
                    # [GSR] OSC: 0.01 < x < 40, phi = 0.99, max = 3.4, fs = 4 Hz
                    # [TMP] OSC: 25 < x < 33.5, min_max old = 28, 30, fs = 4 Hz
                    # [IBI] OSC: 0.5s < x < 1.2s == 120 BPM < x < 50 BPM

                    if len(l) > 0:
                        if biometric == 'pressure':
                            ## modifcation made 11/10 changing mean to median
                            return escalado_11(escalado_minmax(sqrt(median([x**2 if x**2 < 100**2 else 100**2 for x in l])), 0, 100))
                        elif biometric == 'electricity':
                            return escalado_11(escalado_minmax(log(mean(l)+0.99), 0, 3.4))
                        elif biometric == 'temperature':
                            return escalado_11(escalado_minmax(mean(l), 25, 33.5))
                        elif biometric == 'heartRate':
                            return escalado_11(escalado_minmax(60/mean(l), 50, 120))
                    else:
                        return last_values[biometric]
                
                osc_dict = {'pressure': process_value(l_bvp, 'pressure'),
                            'electricity': process_value(l_gsr, 'electricity'),
                            'temperature': process_value(l_tmp, 'temperature'),
                            'heartRate': process_value(l_ibi, 'heartRate')}
                
                for biometric, data in osc_dict.items():
                    client.send_message("/" + biometric, data)
                    client2.send_message("/" + biometric, data)
                    last_values[biometric] = data

                # We get the mean of the temperature and append them to the final array.
                # Temp_tuple = (datetime.now().isoformat(), np.mean(Temporal_Temp_array))
                # Temp_array.append(Temp_tuple)

                # We pause the acquisition of signals for one second
                # time.sleep(3)
            except socket.timeout:
                print("Socket timeout")
                reconnect()
        except KeyboardInterrupt:          
            """
            #Debugging print variables
            print(BVP_array)
            print("*********************************************")
            print()
            print(Acc_array)
            print("*********************************************")
            print()
            print(GSR_array)
            print("*********************************************")
            print()
            print(Temp_array)
            print()
            """
            #print("Disconnecting from device")
            #s.send("device_disconnect\r\n".encode())
            #s.close()
    #stream()

    # MATPLOTLIB'S FIGURE AND SUBPLOTS SETUP
    """
    Gridspec is a function that help's us organize the layout of the graphs,
    first we need to create a figure, then assign a gridspec to the figure.
    Finally create the subplots objects (ax's) assigning a format with gs (gridspec).
    """
    fig = plt.figure(constrained_layout = True)
    gs = fig.add_gridspec(5,1)
    ax1 = fig.add_subplot(gs[0,0])
    ax1.set_title("Temperature")
    ax2 = fig.add_subplot(gs[1,0])
    ax2.set_title("Electrodermal Activity")
    ax3 = fig.add_subplot(gs[2,0])
    ax3.set_title("Blood Volume Pulse")
    ax4 = fig.add_subplot(gs[3,0])
    ax4.set_title("IBI")
    ax5 = fig.add_subplot(gs[4,0])
    ax5.set_title("Fast Fourier Transform")

    # Animation function: this function will update the graph in real time,
    # in order for it to work properly, new data must be collected inside this function.
    def animate(frame):
        global BVP_array
        global GSR_array
        global Temp_array
        global IBI_array
        global Temporal_BVP_array
        global Temporal_GSR_array
        global Temporal_Temp_array
        global Temporal_IBI_array
        global counter
        stream() # This is the function that connects to the Empatica.
        
        #x_BVP_val = np.linspace(0,len(Temporal_BVP_array)-1,num= len(Temporal_BVP_array))
        #x_GSR_val = np.linspace(0,len(Temporal_GSR_array)-1,num= len(Temporal_GSR_array))
        #x_Temp_val = np.linspace(0,len(Temporal_Temp_array)-1,num= len(Temporal_Temp_array))
        #x_IBI_val = np.linspace(0,len(Temporal_IBI_array)-1,num= len(Temporal_IBI_array))
        
        x_BVP_val = np.arange(0.015625,((len(Temporal_BVP_array))*0.015625)+0.015625,0.015625)
        x_GSR_val = np.arange(0.25,((len(Temporal_GSR_array))*0.25)+0.25,0.25)
        x_Temp_val = np.linspace(0,len(Temporal_Temp_array)-1,num= len(Temporal_Temp_array))
        x_IBI_val = np.linspace(0,len(Temporal_IBI_array)-1,num= len(Temporal_IBI_array))
        
        X = rfft(Temporal_BVP_array)
        xf = rfftfreq(len(Temporal_BVP_array), 1/64)

        # GRAPHING ASSIGNMENT SECTION
        # First the previous data must be cleaned, then we plot the array with the updated info.
        ax1.clear()
        ax2.clear()
        ax3.clear()
        ax4.clear()
        ax5.clear()
        ax1.set_ylim(25,33.5) #We fixed the y-axis values to observe a better data representation.
        #ax2.set_ylim(0, 0.5)
        ax3.set_ylim(-150,150)
        ax4.set_ylim(50,120)
        ax1.set_title("Temperature")
        ax2.set_title("Electrodermal Activity")
        ax3.set_title("Blood Volume Pulse")
        ax4.set_title("IBI")
        ax5.set_title("Fast Fourier Transform")
        ax1.set_ylabel("Celsius (°C)")
        ax2.set_ylabel("Microsiemens (µS)")
        ax3.set_ylabel("Nano Watt")
        ax4.set_ylabel("Beats Per Minute (BPM)")
        ax5.set_ylabel("Magnitude")
        ax1.set_xlabel("Samples")
        ax2.set_xlabel("Seconds")
        ax3.set_xlabel("Seconds")
        ax4.set_xlabel("Samples")
        ax5.set_xlabel("Frequency (Hz)")

        if (counter >= 2400):
            ax1.plot(x_Temp_val,Temporal_Temp_array, color = "#F1C40F")
            ax2.plot(x_GSR_val[-200:],Temporal_GSR_array[-200:], color = "#16A085")
            ax3.plot(x_BVP_val[-2000:],Temporal_BVP_array[-2000:])
            ax4.plot(x_IBI_val, Temporal_IBI_array, color = '#F2220C')
            ax5.plot(xf, np.abs(X))

        else:
            ax1.plot(x_Temp_val,Temporal_Temp_array, color = "#F1C40F")
            ax2.plot(x_GSR_val,Temporal_GSR_array, color = "#16A085")
            ax3.plot(x_BVP_val,Temporal_BVP_array)
            ax4.plot(x_IBI_val, Temporal_IBI_array, color = '#F2220C')
            ax5.plot(xf, np.abs(X))

        counter += 60  

    # Here es where the animation is executed. Try encaspsulation allows us
    # to stop the code anytime with Ctrl+C.
    try:
        anim = animation.FuncAnimation(fig, animate,
                                    frames = 500, 
                                    interval = 1000)
        # Once the Animation Function is ran, plt.show() is necesarry, 
        # otherwise it won't show the image. Also, plt.show() will stop the execution of the code 
        # that is located after. So if we want to continue with the following code, we must close the 
        # tab generated by matplotlib.   
        plt.show()


        # The next lines allow us to create a CSV file with data retrieved from E4 wristband.
        # This code is repeated if a KeyboardInterrupt exception arises as a redundant case
        # for storing the data recorded.
        with open("{}/Raw/fileBVP.csv".format(folder), 'w', newline = '') as document:
            writer = csv.writer(document)
            writer.writerow(['Datetime', 'valueBVP'])
            writer.writerows(BVP_array)

        """ with open("{}/Raw/fileACC.csv".format(folder), 'w', newline = '') as document:
            writer = csv.writer(document)
            writer.writerow(['Datetime', 'valueACC'])
            writer.writerows(Acc_array) """   

        with open("{}/Raw/fileEDA.csv".format(folder), 'w', newline = '') as document:
            writer = csv.writer(document)
            writer.writerow(['Datetime', 'valueEDA'])
            writer.writerows(GSR_array)        

        with open("{}/Raw/fileTemp.csv".format(folder), 'w', newline = '') as document:
            writer = csv.writer(document)
            writer.writerow(['Datetime', 'valueTemp'])
            writer.writerows(Temp_array)        

        with open("{}/Raw/fileIBI.csv".format(folder), 'w', newline = '') as document:
            writer = csv.writer(document)
            writer.writerow(['Datetime', 'valueIBI'])
            writer.writerows(IBI_array)   

        # These next instructions should be executed only once, and exactly where we want the program to finish.
        # Otherwise, it may rise a Socket Error. These lines also written below in case of a KeyBoardInterrupt 
        # exception arising.
        global s
        print("Disconnecting from device")
        s.send("device_disconnect\r\n".encode())
        s.close()

    except KeyboardInterrupt:
        print('hola')
        #The next lines allow us to create a CSV file with data retrieved from E4 wristband.
        with open("{}/Raw/fileBVP.csv".format(folder), 'w', newline = '') as document:
            writer = csv.writer(document)
            writer.writerow(['Datetime', 'valueBVP'])
            writer.writerows(BVP_array)

        """ with open("{}/Raw/fileACC.csv".format(folder), 'w', newline = '') as document:
                    writer = csv.writer(document)
                    writer.writerow(['Datetime', 'valueACC'])
                    writer.writerows(Acc_array) """   

        with open("{}/Raw/fileEDA.csv".format(folder), 'w', newline = '') as document:
                    writer = csv.writer(document)
                    writer.writerow(['Datetime', 'valueEDA'])
                    writer.writerows(GSR_array)        

        with open("{}/Raw/fileTemp.csv".format(folder), 'w', newline = '') as document:
                    writer = csv.writer(document)
                    writer.writerow(['Datetime', 'valueTemp'])
                    writer.writerows(Temp_array)        

        with open("{}/Raw/fileIBI.csv".format(folder), 'w', newline = '') as document:
                    writer = csv.writer(document)
                    writer.writerow(['Datetime', 'valueIBI'])
                    writer.writerows(IBI_array)   

        # We close connections
        print("Disconnecting from device")
        s.send("device_disconnect\r\n".encode())
        s.close()

# # CODE FOR EEG # #
def EEG(seconds,folder,ip,port1,port2,port3,port4,port5,client1,client2
    ,client3,client4,client5,arduino_1,arduino_bol,synthetic_bol):
    global simulation_time
    ########################################################################################
    #Variable a modificar

    # Define the function to compute PSD for multiple frequency bands
    final_descion=int(input('0 for only emotions; 1 for only fear; 2 for both:'))
    Val_Pkl = pickle.load(open('Val_RF_10s.pkl', 'rb'))
    Aro_Pkl = pickle.load(open('Aro_RF_10s.pkl', 'rb'))
    Dom_Pkl = pickle.load(open('Dom_RF_10s.pkl', 'rb'))

    iteraciones = 0
    warnings.filterwarnings("ignore", category=UserWarning, module="sklearn")
    warnings.filterwarnings("ignore", category=DataConversionWarning)

    # Set the duration to stream data (5 seconds in this example)
    duration = 10 # queremos 10 segundos
    # Set the sampling rate and channel(s) you want to stream

    ## SEPTEMBER 28 UPDATE ##
    sampling_rate = 250 # used to be 128, queremos 250 Hz

    #channels = (0, 1, 2, 3, 4, 5, 6, 7)  # Streaming data from channels 0 to 7
    channels = list(range(9))   # Streaming data from channels 0 to 7\n",
    #Path para dicionarios de emociones
    base_path = Path(__file__).parent
    emotions_path=base_path/'Emotions'/"emociones.json"
    fear_path=base_path/"Emotions"/"fear.json"
    ###############################################################################
    def compute_psd_bands(data, fs):
        # Define the frequency ranges for each band
        bands = {
            'Delta': (0.5, 4),
            'Theta': (4, 8),
            'Alpha': (8, 12),
            'Beta': (12, 30),
            'Gamma': (30, 45)}
    
    # Compute the PSD for each frequency band
        psd_bands = {}
        for band, (f_min, f_max) in bands.items():
            power = pe.bin_power(data, [f_min, f_max], fs)
            psd_bands[band]=np.mean(power)
    
        return psd_bands
    def butter_bandpass_filter(data, lowcut, highcut, fs, order=5):
        nyquist = 0.5 * fs
        low = lowcut / nyquist
        high = highcut / nyquist
        b, a = butter(order, [low, high], btype='band')
        y = lfilter(b, a, data)
        return y
    
    with open(emotions_path) as f:
        emotions=json.load(f)

    with open(fear_path) as r:
        emotions_fear=json.load(r)

    def emocion(arousal, dominance, valence, emotions, emotions_fear):
        valence_1 = int(valence - 1)
        arousal_1 = int(arousal - 1)
        dominance_1 = int(dominance - 1)
        # Create the key and ensure it's formatted as a string
        key = str([arousal_1, dominance_1, valence_1])
    
        if final_descion == 0:  # Only emotions
            return emotions.get(key, 'Unknown')
        elif final_descion == 1:  # Only fear
            return emotions_fear.get(key, 'Unknown')
        elif final_descion == 2:  # Both emotions and fear
            emotion_value = emotions.get(key, 'Unknown')
            fear_value = emotions_fear.get(key, 'Unknown')
            return emotion_value, fear_value
        else:
            print("error")
            return None
    def real_emotion(emo):
        map_emotions = {"Sadness": "Sadness",
                        "Rejected": None,
                        "Pessimistic": None,
                        "Hate": "Hate",
                        "Distressed": None, 
                        "Anxious": None,
                        "Calm": None,
                        "Neutral": None,
                        "Admiration": "Admiration",
                        "Relief": None,
                        "Relaxed": None,
                        "Overconfident": None,
                        "Satisfied": None,
                        "Desire": "Desire",
                        "Love": "Love",
                        "Joy": "Joy",
                        "Generosity": None }
    #emo = map_emotions[emo]
    
        if emo == "Love":
            return 1
        elif emo == "Hate":
            return 2
        elif emo == "Desire":
            return 3
        elif emo == "Admiration":
            return 4
        elif emo == "Joy":
            return 5
        elif emo == "Sadness":
            return 6

        #ESPAÑOL
        elif emo == "No Fear":
            return 0.0
        elif emo == "Low Fear":
            return 0.33
        elif emo == "Medium Fear":
            return 0.66
        elif emo == "High Fear":
            return 1
        else: 
            return 0
        
    def setup_arduino():
        arduino_com = input("What Number of COM is your Arduino: ")
        arduino = Serial(port='COM' + arduino_com, baudrate=115200, timeout=.1)
        print(f"Arduino connected on COM{arduino_com}.")
    
        def write_read(x, y, z):
            arduino.write(bytes(str(x), 'utf-8'))
            time.sleep(0.05)
            arduino.write(bytes(str(y), 'utf-8'))
            time.sleep(0.05)
            arduino.write(bytes(str(z), 'utf-8'))
            time.sleep(0.05)
    
        return arduino, write_read

# Función para manejar datos sintéticos u reales (OpenBCI)
    def setup_board(is_synthetic):
        params = BrainFlowInputParams()
        if is_synthetic:
            board = BoardShim(BoardIds.SYNTHETIC_BOARD.value, params)
            print("Using synthetic values...")
        else:
            open_bci_com = input("What number of COM is the BCI: ")
            params.serial_port = 'COM' + open_bci_com
            board = BoardShim(BoardIds.CYTON_BOARD.value, params)
            print(f"BCI connected on COM{open_bci_com}.")
        board.prepare_session()
        imestamp_channel = board.get_timestamp_channel(BoardIds.CYTON_BOARD.value if not is_synthetic else BoardIds.SYNTHETIC_BOARD.value)
        acc_channel = board.get_accel_channels(BoardIds.CYTON_BOARD.value if not is_synthetic else BoardIds.SYNTHETIC_BOARD.value)
        return board, timestamp_channel, acc_channel
    
    #   1. Sí Arduino, Sí Sintético
    if arduino_bol and synthetic_bol:
        print("Using both Arduino and Synthetic data.")
        arduino, write_read = setup_arduino()
        board, timestamp_channel, acc_channel = setup_board(True)

    # 2. Sí Arduino, No Sintético
    elif arduino_bol and not synthetic_bol:
        print("Using Arduino and Real BCI data.")
        arduino, write_read = setup_arduino()
        board, timestamp_channel, acc_channel = setup_board(False)

    # 3. No Arduino, Sí Sintético
    elif not arduino_bol and synthetic_bol:
        print("Using only Synthetic data (no Arduino).")
        board, timestamp_channel, acc_channel = setup_board(True)

    # 4. No Arduino, No Sintético
    else:
        print("Using only Real BCI data (no Arduino, no Synthetic).")
        board, timestamp_channel, acc_channel = setup_board(False)


    count = 0
    repetitions = 0

    escalado_11 = lambda x: (x - 0.5)*2  ## CAMBIAR 0.5 POR VALOR NEUTRO DE ENGAGEMENT??

    subject_ID, repetition_num = input('Please enter the subject ID and the number of repetition: ').split(' ')
    subject_ID = '0' + subject_ID if int(subject_ID) < 10 else subject_ID
    repetition_num = '0' + repetition_num if int(repetition_num) < 10 else repetition_num
    lodb = base_path
    folder = f'S{subject_ID}R{repetition_num}_{datetime.datetime.now().strftime("%d%m%Y_%H%M")}'
    os.mkdir(f"{lodb}/{folder}")

    df_eeg = pd.DataFrame()
    df_time = pd.DataFrame()
    df_emotions = pd.DataFrame(data = [], columns=['Timestamp', 'Engagement', 'Emotion', 'Valence', 'Arousal', 'Dominance'])
    df_acc = pd.DataFrame() # acceleration
    df_fear=pd.DataFrame(data = [], columns=['Timestamp', 'Engagement', 'Emotion', 'Valence', 'Arousal', 'Dominance'])

    print(Fore.RED + 'Initializing functions...' + Style.RESET_ALL)
    

    try:   
     while iteraciones < 2000:

        # Create empty lists to store the streamed data for each channel
        channel_data = [[] for _ in channels]
        channel_data_acc = [[] for _ in acc_channel] # acceleration

        # Start the streaming
        board.start_stream()

        # Get the start time
        start_time = time.time()

        # Loop until the specified duration is reached
        while time.time() - start_time < duration:
            # Fetch the latest available samples
            samples = board.get_current_board_data(sampling_rate)

            # Append the samples to the corresponding channel's data list
            for i, channel in enumerate(channels):
                channel_data[i].extend(samples[channel])

            np_time = np.array(samples[timestamp_channel])
            np_time = np_time - 21600 # time zone converter to GMT-6
            np_df = pd.DataFrame(np_time)
            df_time = pd.concat([df_time, np_df], ignore_index=True)
            
            
            ## ACCELERATION ##
            
            for i, channel in enumerate(acc_channel):
                channel_data_acc[i].extend(samples[channel])
                                

            # Sleep for a small interval to avoid high CPU usage
            time.sleep(1)

        # Stop the streaming
        board.stop_stream()

            # acceleration dataframe
        data_dict_acc = {f'Channel_{channel}': channel_data_acc[i] for i, channel in enumerate(acc_channel)}
        df_acc_prueba = pd.DataFrame(data_dict_acc)
        df_acc = pd.concat([df_acc, df_acc_prueba], ignore_index=True)


        # Create a dictionary with channel names as keys and data as values
        data_dict = {f'Channel_{channel}': channel_data[i] for i, channel in enumerate(channels)}

        # Create a DataFrame from the data dictionary
        df = pd.DataFrame(data_dict)

        row_all_zeros = (df == 0).all(axis=1)
        df2 = df[~row_all_zeros]
        df3 = df2.drop(df.columns[0], axis=1)
        df4 = df3[['Channel_1', 'Channel_2', 'Channel_3', 'Channel_4', 'Channel_5', 'Channel_6', 'Channel_7', 'Channel_8']].copy()
        
        
        # Append data to global dataframe
        df_eeg = pd.concat([df_eeg, df4], ignore_index=True)
                
        lowcut = 0.4  # Lower cutoff frequency in Hz
        highcut = 45  # Upper cutoff frequency in Hz
        fs = 128  # Sampling rate in Hz

        ratio = 128/250
        df5 = df4.iloc[::int(1/ratio)].interpolate()

        # Apply the bandpass filter to each column
        filtered_df = df5.apply(lambda col: butter_bandpass_filter(col, lowcut, highcut, fs))

        average_reference = filtered_df.mean(axis=1)
        df_average_reference = filtered_df.sub(average_reference, axis=0)

          # Iterate over each column in your DataFrame
        for column in df_average_reference.columns:
            # Compute the PSD for the column data and frequency bands
            psd_bands = compute_psd_bands(df_average_reference[column].values, fs=128)
        
            # Add the PSD values to the DataFrame
            psd_df = pd.concat([psd_df, pd.DataFrame([psd_bands])], ignore_index=True)


        df_t = psd_df.transpose()
        df_t.columns = ['Fp1', 'Fp2', 'C3', 'C4', 'P7', 'P8', 'O1', 'O2']

        df_t = df_t.reset_index()

        # Use the melt function to reshape the DataFrame
        melted_df = pd.melt(df_t, id_vars='index', var_name='channel', value_name='value')

        # Convert channel numbers to strings
        melted_df['channel'] = melted_df['channel'].astype(str)

        # Create a new 'channel_band' column by combining 'channel' and 'index' columns
        melted_df['channel_band'] = melted_df['channel'] + '_' + melted_df['index']

        # Pivot the DataFrame to get the desired format
        new_df = melted_df.pivot(index='index', columns='channel_band', values='value')

        series = new_df.stack()

        # Convert the Series back to a DataFrame with a single row
        filter_df = pd.DataFrame(series)

        valo =filter_df[0]
        valores = valo.reset_index(drop=True)
        df_modelo = pd.DataFrame(valores).transpose()

        df_modelo.columns = ['Fp1_Delta', 'Fp1_Theta', 'Fp1_Alpha','Fp1_Beta','Fp1_Gamma',
                             'Fp2_Delta', 'Fp2_Theta', 'Fp2_Alpha','Fp2_Beta','Fp2_Gamma',
                             'C3_Delta', 'C3_Theta', 'C3_Alpha','C3_Beta','C3_Gamma',
                             'C4_Delta', 'C4_Theta', 'C4_Alpha','C4_Beta','C4_Gamma',
                             'P7_Delta', 'P7_Theta', 'P7_Alpha','P7_Beta','P7_Gamma',
                             'P8_Delta', 'P8_Theta', 'P8_Alpha','P8_Beta','P8_Gamma',
                             'O1_Delta', 'O1_Theta', 'O1_Alpha','O1_Beta','O1_Gamma',
                             'O2_Delta', 'O2_Theta', 'O2_Alpha','O2_Beta','O2_Gamma',]
        
        df_pred = df_modelo.reset_index(drop=True)

        CANALES = ['Fp1', 'Fp2', 'C3', 'C4', 'P7', 'P8', 'O1', 'O2']

        for channel in CANALES:
            df_pred[f'{channel}_Engagement'] = df_pred[f'{channel}_Beta'] / (df_pred[f'{channel}_Theta'] + df_pred[f'{channel}_Alpha'])

        for channel in CANALES:
            df_pred[f'{channel}_Fatigue'] = df_pred[f'{channel}_Alpha'] / df_pred[f'{channel}_Theta']

        for channel in CANALES:
            df_pred[f'{channel}_Excitement'] = df_pred[f'{channel}_Beta'] / df_pred[f'{channel}_Alpha']

        for channel in CANALES:
            df_pred[f'{channel}_Relaxation'] = df_pred[f'{channel}_Theta'] / df_pred[f'{channel}_Delta']


        iteraciones += 1

        valen = Val_Pkl.predict(df_pred)
        arous = Aro_Pkl.predict(df_pred)
        domin = Dom_Pkl.predict(df_pred)
                 
        engag_fp1 = mean(df_pred["Fp1_Engagement"])
        engag_fp2 = mean(df_pred["Fp2_Engagement"])
        engag_c3 = mean(df_pred["C3_Engagement"])
        engag_c4 = mean(df_pred["C4_Engagement"])
        engag_p7 = mean(df_pred["P7_Engagement"])
        engag_p8 = mean(df_pred["P8_Engagement"])
        engag_o1 = mean(df_pred["O1_Engagement"])
        engag_o2 = mean(df_pred["O2_Engagement"])

         ## LINEAS ACTUALES - SE CAMBIARON EL 25 DE AGOSTO DE 2023 ##
        ##engagement = ((engag_fp1+engag_fp2+engag_c3+engag_c4+engag_p7+engag_p8+engag_o1+engag_o2)/8)
        engagement = ((engag_fp1+engag_fp2)/2)
        engag = escalado_11(engagement)
        
        #engag = math.log((engag_fp1+engag_fp2)/2) #se agregó el logaritmo
        #engag = engag/2                           #se divide sobre 2 (rangos aprox de -2 a 2), con los IF, se limita a -1 y 1
        #if engag <= 0:
        #    engag = 0
        #if engag >=1:
        #    engag = 1

        ## LINEAS ANTERIORES -  CAMBIARON EL 25 DE AGOSTO DE 2023 ##
        #engag = ((engag_fp1+engag_fp2)/2*10) #se agregó una multiplicación 
        #engag = escalado_11(engag)
        
        vale = mean(valen)
        arou = mean(arous)
        domi = mean(domin)
        print(vale, arou, domi) #descomentar
        emociones = emocion(arou, domi, vale,emotions,emotions_fear)
        if final_descion == 0:  # Solo emociones
                print(f"Emotion: {emociones}")
        elif final_descion == 1:  # Solo miedo
            print(f"Fear: {emociones}")
        elif final_descion == 2:  # Ambos
            emotion_value, fear_value = emociones
            print(f"Emotion: {emotion_value}, Fear: {fear_value}")

          #write_read(vale, arou, domi)
        values = [vale, arou, domin]
        realemotion = real_emotion(emociones)
        if final_descion == 0:  # Only emotions
            client1.send_message(address, engag)
            client2.send_message(address, engag)
        elif final_descion == 1:  # Only fear
            client3.send_message(address2, emociones)
            client4.send_message(address3, realemotion)
        elif final_descion == 2:  # Both emotions and fear
            emotion_value, fear_value = emociones
            client1.send_message(address, engag)
            client2.send_message(address, engag)
            client3.send_message(address2, emotion_value)
            client4.send_message(address3, realemotion)
            client5.send_message(address2, fear_value)

        if final_descion == 0:
            df_emotions.loc[len(df_emotions)] = [time.time() - 21600, engag, emociones, vale, arou, domi]
        elif final_descion == 1:
            df_fear.loc[len(df_fear)] = [time.time() - 21600, engag, emociones, vale, arou, domi]
        elif final_descion == 2:
            df_emotions.loc[len(df_emotions)] = [time.time() - 21600, engag, emociones[0], vale, arou, domi]
            df_fear.loc[len(df_fear)] = [time.time() - 21600, engag, emociones[1], vale, arou, domi]


        df_emotions.style.format("{:.2f}")   
        print(emociones) #descomentar
        print(engag) #descomentar 
        df_fear.style.format("{:.2f}")  
        # print(fear_cal)

    except KeyboardInterrupt:     
        print(Fore.BLUE + 'Test interrupted. Storing data...' + Style.RESET_ALL)        

        df_eeg = df_eeg.reset_index(drop=True)
        df_time = df_time.reset_index(drop=True)  
        df_time.columns = ['Timestamp']
        df_emotions = df_emotions.reset_index(drop=True)
        df_fear=df_fear.reset_index(drop=True)  
    
    
        df_acc = df_acc.reset_index(drop=True) 
        df_acc.columns = ['Acc_1', 'Acc_2', 'Acc_3']     
       
         # acceleration
         #print(df_acc.shape)
        #print(df_eeg.shape)  
        df_complete = pd.concat([df_time, df_eeg, df_acc], axis=1)
        df_complete = df_complete.reset_index(drop=True)  
      
        #df_eeg.to_csv('{}/EEG.csv'.format(folder), mode='a')
        #df_time.to_csv('{}/TimeStamps.csv'.format(folder), mode='a') 
    
        if final_descion == 0:
            path_1 = f'{lodb}/{folder}/S{subject_ID}R{repetition_num}_Emotions.csv'
            print(f'Guardando emociones en: {path_1}')
            df_emotions.to_csv(path_1, mode='a')

        elif final_descion == 1:
            path_2 = f'{lodb}/{folder}/S{subject_ID}R{repetition_num}_Fear.csv'
            print(f'Guardando miedo en: {path_2}')
            df_fear.to_csv(path_2, mode='a')

        elif final_descion == 2:
            path_1 = f'{lodb}/{folder}/S{subject_ID}R{repetition_num}_Emotions.csv'
            path_2 = f'{lodb}/{folder}/S{subject_ID}R{repetition_num}_Fear.csv'
            print(f'Guardando emociones en: {path_1}')
            print(f'Guardando miedo en: {path_2}')
            df_emotions.to_csv(path_1, mode='a')
            df_fear.to_csv(path_2, mode='a')

        #df_acc.to_csv('{}/Acceleration.csv'.format(folder), mode='a') # acceleration
        print(f'Emotions{df_emotions}')
        print(f'Fear:{df_fear}')
    
        #ldoa=input("directorio de tu carpeta de brain: ")

        #oped=pd.read_csv(path_1)
        #print(oped)
        board.stop_stream()
        board.release_session()
        print(Fore.GREEN + 'Data stored.' + Style.RESET_ALL)   
        import pandas as pd

        if final_descion == 0 and path_1:
            table_1 = pd.read_csv(path_1).tail(20)
            combined_table = table_1
        elif final_descion == 1 and path_2:
            table_2 = pd.read_csv(path_2).tail(20)
            combined_table = table_2
        elif final_descion == 2 and path_1 and path_2:
            table_1 = pd.read_csv(path_1).tail(20)
            table_2 = pd.read_csv(path_2).tail(20)
            combined_table = pd.concat([table_1, table_2])

        num_columns = ['Timestamp', 'Engagement', 'Valence', 'Arousal', 'Dominance']
    styled_table = (
        combined_table.style
        .format({col: "{:.2f}" for col in num_columns if col in combined_table.columns})  # Formatear solo columnas numéricas
        .highlight_max(subset=['Engagement'], axis=0, color='pink')  
        .highlight_min(subset=['Engagement'], axis=0, color='blue')  
    )


    cell_hover = {'selector': 'td:hover', 'props': [('background-color', '#ffffb3')]}
    headers = {'selector': 'th:not(.index_name)', 'props': [('background-color', '#00FF00'), ('color', 'black')]}
    index_names = {'selector': '.index_name', 'props': [('font-style', 'italic'), ('color', 'darkgrey'), ('font-weight', 'normal')]}

    styled_table = (
        styled_table
        .set_table_styles([cell_hover, index_names, headers])
        .set_table_styles(
            [{'selector': 'th', 'props': [('border', '1px solid black')]},  
             {'selector': 'td', 'props': [('border', '1px solid black')]}])
    )

    with open(f'{lodb}/{folder}/styled_table.html', 'w') as f:
        f.write(styled_table.render())

    print("Table saved as 'styled_table.html'")
    
    ##############Links que pueden ayudar al entendimiento del código ##############
    # https://www.geeksforgeeks.org/python-save-list-to-csv/

def MachineLearning(second, folder):

    # Pipeline for normalization:
    # 1. Remove outliers based on quantile method (1 < second < 91)
    # 2. Combine features
    # 3. Normalize using minmax scaler (61 < second < 91)
    
    fatigue_fig = plt.figure(4, constrained_layout=True)
    ax_fatigue = fatigue_fig.add_subplot(111)
    ax_fatigue.plot([-1], [-1])
    ax_fatigue.set_title('Fatigue level (FAS)')
    ax_fatigue.set_ylim([0, 50])
    ax_fatigue.set_xlim([0, 1])
    ax_fatigue.set_xlabel('Samples')
    ax_fatigue.set_ylabel('Fatigue score')
    plt.pause(1)

    scaler = None
    emp = False # Indica si utiliza la empatica o no
    free_flag = True
    a_prediction = False
    current_second = 0
    l_fatigue = []

    name_file = 'EEG_Empatica' if emp else 'EEG'

    model = load(open('Empatica-Project-ALAS-main/Files/model_{}.pkl'.format(name_file), 'rb'))
    file_features = open('Empatica-Project-ALAS-main/Files/features_{}.txt'.format(name_file), 'r')
    features = file_features.readlines()[0].split(', ')
    file_features.close()

    operations = ['D', 'I', 'L', 'M']
    spectral_signals = ['Alpha', 'Beta', 'Gamma', 'Delta', 'Theta']
    empatica_variables = ['BVP', 'Temp', 'ACC', 'IBI', 'EDA']

    all_features = list(set([x for y in features for x in y.split('-') if x not in operations]))
    empatica_features = [x for x in all_features if x.split('_')[0] in empatica_variables]
    eeg_features = [x for x in all_features if x.split('_')[0] in spectral_signals]

    cv = lambda x: np.std(x, ddof=1) / np.mean(x) * 100  # Coefficient of variation
    funcs_dict = dict(zip(['Mean', 'StandardDeviation', 'Median', 'Maximum', 'Minimum',
                            'InterquartileRange', 'Kurtosis', 'Skewness', 'CoefficientVariation'],
                            [np.mean, np.std, np.median, np.max, np.min, stats.iqr, stats.kurtosis, stats.skew, cv]))

    funcs_names = [x.split('_')[0] for x in empatica_features]
    agg_funcs = [funcs_dict[x] for x in funcs_names]

    def combine_feature(df, feature_name):
        if len(feature_name.split('-')) == 1:
            return np.array(df[feature_name])

        operation = feature_name.split('-')[1]
        variable1 = feature_name.split('-')[0]

        if operation == 'D':
            variable2 = feature_name.split('-')[2]
            return np.array(np.divide(np.array(df[variable1]), np.array(df[variable2])))
        elif operation == 'M':
            variable2 = feature_name.split('-')[2]
            return np.array(np.multiply(np.array(df[variable1]), np.array(df[variable2])))
        elif operation == 'I':
            return np.array(np.divide(np.ones(df.shape[0]), np.array(df[variable1])))
        elif operation == 'L':
            return np.array(np.log(np.array(np.array(df[variable1])) + 1))
        
    def get_df(seconds):
        df_calibration = pd.DataFrame()
        for df_name in list(set([x.split('_')[0] for x in all_features])):
            if not df_name.isupper():
                # EEG Features
                df_raw = pd.read_csv('{}/Raw/{}.csv'.format(folder, df_name), index_col=0)

                df_processed_temp = df_raw.apply(pd.to_numeric, errors='coerce').dropna(axis=0)
                # .reset_index(drop=True)

                df_calibration = pd.concat([df_calibration, df_processed_temp], axis=1)
            elif df_name.isupper():
                # Empatica Features
                df = pd.read_csv('{}/Raw/file{}.csv'.format(folder, df_name))
                df['Datetime'] = pd.to_datetime(df.Datetime).apply(datetime.timestamp)
                df['Datetime'] = df.Datetime - df.loc[0, 'Datetime']
                df['Second'] = df.Datetime.apply(floor)
                df['Bins'] = pd.cut(df['Second'], bins=[0] + list(df_calibration.index),
                                    duplicates='drop', include_lowest=True)

                df = df.drop(['Second', 'Datetime'], axis=1).groupby('Bins')
            
                df = df.agg(agg_funcs)['value{}'.format(df_name.split('.')[0][4:])].reset_index(drop=True)
                df.columns = [df_name.split('.')[0][4:] + func_name for func_name in funcs_names]
                df = df.replace(0, np.nan).interpolate(method='polynomial', order=2,  axis=0, limit_direction='forward',
                                                    limit_area='inside')
                df = pd.DataFrame(KNNImputer(n_neighbors=4).fit_transform(np.array(df)), columns=df.columns)

                df_emp = pd.concat([df_emp, df], axis=1)
        
        eeg_idx = df_calibration.index.rename('Second')
        df_calibration.index = eeg_idx
        df_calibration = df_calibration.reset_index()

        df_calibration = df_calibration[eeg_features + ['Second']]
        if emp:
            df_emp['Second'] = eeg_idx
            df_emp = df_emp[empatica_features + ['Second']]
            df_calibration = pd.merge(df_calibration, df_emp, on='Second')

        df_calibration = df_calibration[(df_calibration.Second > seconds[0]) & (df_calibration.Second < seconds[1])]
        return df_calibration

    while True:
        with second.get_lock():
            if(second.value == 332):
                return
            if (second.value > 91) and (second.value < 95) and (scaler is None) and (free_flag):
                free_flag = False
                df_calibration = get_df([31, 91])
                seconds_col = df_calibration.Second
                df_calibration = df_calibration.drop('Second', axis=1)
                calib_Q = [df_calibration.quantile(q=.25), df_calibration.quantile(q=.75), 
                           df_calibration.apply(stats.iqr)] # q1, q3, iqr
                
                df_calibration['Second'] = seconds_col
                df_calibration = df_calibration[(df_calibration.Second > 61) & (df_calibration.Second < 91)]

                df_calibration_combined = pd.DataFrame()
                for feature in features:
                    df_calibration_combined = pd.concat([df_calibration_combined,
                                                        pd.DataFrame(combine_feature(df_calibration, feature), columns=[feature])],
                                                        ignore_index=True, axis=1)
                df_calibration_combined.columns = features
                
                scaler = MinMaxScaler().fit(df_calibration_combined)
                free_flag = True

            if (second.value > 91) and (second.value % 10 == 0) and (scaler is not None) and (current_second != second.value):
                current_second = second.value
                df_current = get_df([second.value - 10, second.value]).drop('Second', axis=1)
                df_current = df_current[~((df_current < (calib_Q[0] - 1.5 * calib_Q[2])) | (df_current > (calib_Q[1] + 1.5 * calib_Q[2]))).any(axis=1)]
                
                if df_current.shape[0] > 0:
                    df_current_combined = pd.DataFrame(columns=features)
                    for feature in features:
                        df_current_combined[feature] = combine_feature(df_current, feature)

                    df_current_combined_transformed = pd.DataFrame(scaler.transform(df_current_combined), columns=features)
                    raw_prediction = model.predict(df_current_combined_transformed)
                    prediction = np.median(np.round(raw_prediction)) if abs(np.mean(raw_prediction)) > 50 else round(np.mean(raw_prediction))
                    a_prediction = True

                    print(second.value, raw_prediction, prediction)
                    
                    for element in raw_prediction:
                        if element < 0:
                            element = 0
                        elif element > 50:
                            element = 50
                        l_fatigue.append(element)
                else:
                    prediction = np.nan
                df_predictions = pd.Series([prediction], index=[second.value], name='FAS').head(1)
                
                df_predictions.to_csv('{}/predictions.csv'.format(folder), mode='a')

            if (a_prediction) and (second.value > 100) and (second.value % 5 == 0) and (second.value % 10 != 0) and (current_second != second.value) and (scaler is not None):
                current_second = second.value

                s_predictions = pd.Series(data=l_fatigue)
                s_predictions.index = [x + 1 for x in list(s_predictions.index)]
                
                print(s_predictions)

                ax_fatigue.clear()
                ax_fatigue.plot(s_predictions)
                ax_fatigue.set_title('Fatigue level (FAS)')
                ax_fatigue.set_ylim([0, 50])
                ax_fatigue.set_xlim([0, s_predictions.shape[0] + 1])
                ax_fatigue.set_xlabel('Samples')
                ax_fatigue.set_ylabel('Fatigue score')
                plt.pause(1)
                
                a_prediction = False

# Finally, for both processes to run, this condition has to be met. Which is met
# if you run the script.
if __name__ == '__main__':
      
    parser = argparse.ArgumentParser()
    parser.add_argument("--ip", default="10.12.181.191", help="The ip of the OSC server") #10.12.181.191
    parser.add_argument("--port", type=int, default=6000, help="The port the OSC server is listening on")
    args = parser.parse_args()
    client = udp_client.SimpleUDPClient(args.ip, args.port)

    parser2 = argparse.ArgumentParser()
    parser2.add_argument("--ip", default="10.12.181.191", help="The ip of the OSC server") # 10.22.16.247
    parser2.add_argument("--port", type=int, default=6001, help="The port the OSC server is listening on")
    args2 = parser2.parse_args()
    client2 = udp_client.SimpleUDPClient(args2.ip, args2.port)

    # # Create random distribution of beeps represented by ones and zeros # #
    # For the P300 test, a distribution of frequent and non-frequent beeps

    # of 80/20 respectively is needed from a total of 120 beeps.
    # Therefore, 96 frequent and 24 non-frequent beeps are needed. 
    # Zeros will represent frequent beeps and ones non-frequent.
    F = 96
    NF = 24
    distribution_array = np.array([0]*F + [1]*NF) # This array contains 96 zeros and 24 ones
    
    # Now we suffle the distribution array to have a random order
    np.random.shuffle(distribution_array)

    # # Define the data folder # #
    # The name of the folder is defined depending on the user's input
    subject_ID, repetition_num = input('Please enter the subject ID and the number of repetition: ').split(' ')
    subject_ID = '0' + subject_ID if int(subject_ID) < 10 else subject_ID
    repetition_num = '0' + repetition_num if int(repetition_num) < 10 else repetition_num
    folder = 'P300_S{}R{}_{}'.format(subject_ID, repetition_num, datetime.now().strftime("%d%m%Y_%H%M"))
    os.mkdir(folder)

    for subfolder in ['Raw', 'Processed', 'Figures']:
        os.mkdir('{}/{}'.format(folder, subfolder))

    # # Create a multiprocessing List # # 
    # This list will store the seconds where a beep was played
    timestamps = multiprocessing.Manager().list()
    


    #####################################
    ip = "192.168.0.158"  # IP address of the receiving device
    port1 = 9000 
    port2 = 9001
    port5 = 1337 #arduino

    port3 = 9010 # TD miedo
    port4 = 9011 #PD miedo


    client1 = udp_client.SimpleUDPClient(ip, port1)
    client2 = udp_client.SimpleUDPClient(ip, port2)

    client3 = udp_client.SimpleUDPClient(ip, port3)
    client4 = udp_client.SimpleUDPClient(ip, port4)

    client5 = udp_client.SimpleUDPClient(ip, port5) #arduino


    address = "/engagement"  # OSC address to send the message t o
    address2 = "/real-emotion"  # OSC address to send the message to
    address3 = "/emotion-ID"
    address4= "/Fear-emotions"

    arduino_1=input("Do you want to use Arduino? (y/n)").lower()
    arduino_bol=arduino_1=='y'
    synthetic_bol=input("Do you want to use synthetic values? (yes/no): ").lower()=='y'

    #####################################
    # # Start processes # #
    #process1 = Process(
    #    target=beep, args=[seconds, counts, distribution_array, timestamps])
    process2 = Process(
        target=timer, args=[seconds, counts, timestamps])
    p = Process(target=empatica, args=[seconds, folder, client, client2]) #Descomentar para Empatica
    q = Process(target=EEG, args=[seconds,folder,ip,port1,port2,port3,port4,port5,client1,client2
    ,client3,client4,client5,arduino_1,arduino_bol,synthetic_bol])
    m = Process(target=MachineLearning, args=[seconds, folder])
    #process1.start()
    process2.start()
    p.start() # Descomentar para Empatica
    q.start()
    m.start()
    #process1.join()
    process2.join()
    p.join() # Descomentar para Empatica
    q.join()
    m.join()

    # # DATA STORAGE SECTION # #
    # Executed only once the test has finished.
    print(Fore.RED + 'Test finished sucessfully, storing data now...' + Style.RESET_ALL)
    # # Save beeps' timestamps in a .csv file # #
    # We must first convert the multiprocess.Manger.List to a normal list
    timestamps_final = list(timestamps)

    # Now we convert each of the UNIX-type timestamps to normal timestam (year-month-day hour-minute-second-ms)
    for i in range(len(timestamps_final)):
        timestamp = datetime.fromtimestamp(timestamps_final[i])
        timestamps_final[i] = timestamp.strftime('%Y-%m-%d %H:%M:%S.%f')
    
    # Now we transform the data representation from the distribution array, which contains 0 for
    # a frequent beep and 1 for non-frequent beeps. To store it into a CSV, we will store literally
    # the frequent or not-frequent label, instead of 0 and 1 (readibility purpose)
    beeps_distribution = []
    for i in distribution_array:
        if i == 1:
            beeps_distribution.append('Not Frequent')
        else:
            beeps_distribution.append('Frequent')
    # Now we add an extra label which marks the script's initialization timestamp
    # this timestamp will be the first one, always. And the next 2 sounds are 
    # transition beeps, always.
    beeps_distribution.insert(0, 'Start of P300 test')
    beeps_distribution.insert(1, 'Transition Beep')
    beeps_distribution.insert(2, 'Transition Beep')

    # Store data in a .csv
    timestamps_final = pd.Series(timestamps_final)   
    df = pd.DataFrame(timestamps_final, columns=['Beep_Timestamp'])
    df['Timestamp_label'] = pd.Series(beeps_distribution)
    df.to_csv('{}/Timestamps.csv'.format(folder), index=False)
    print(Fore.GREEN + 'Data stored sucessfully' + Style.RESET_ALL)
    
    # # Data processing # #
    print(Fore.RED + 'Data being processed...' + Style.RESET_ALL)


    def remove_outliers(df, method):
        """
        Uses an statistical method to remove outlier rows from the dataset x, and filters the valid rows back to y.

        :param pd.DataFrame df: with non-normalized, source variables.
        :param string method: type of statistical method used.
        :return pd.DataFrame: Filtered DataFrame
        """

        # The number of initial rows is saved.
        n_pre = df.shape[0]

        # A switch case selects an statistical method to remove rows considered as outliers.
        if method == 'z-score':
            z = np.abs(stats.zscore(df))
            df = df[(z < 3).all(axis=1)]
        elif method == 'quantile':
            q1 = df.quantile(q=.25)
            q3 = df.quantile(q=.75)
            iqr = df.apply(stats.iqr)
            df = df[~((df < (q1 - 1.5 * iqr)) | (df > (q3 + 1.5 * iqr))).any(axis=1)]
        
        # The difference between the processed and raw rows is printed.
        n_pos = df.shape[0]
        diff = n_pre - n_pos
        print(f'{diff} rows removed {round(diff / n_pre * 100, 2)}%')
        return df
    
    # The following for loop iterates over all features, and removes outliers depending on the statistical method used.
    # It reads the files saved in the "Raw" folder, and only reads .CSV files, to outputt a .CSV file in "Processed" folder.
    for df_name in os.listdir('{}/Raw/'.format(folder)):
        if df_name[-4:] == '.csv' and df_name[:4] != 'file':
            df_name = df_name[:-4]
            df_raw = pd.read_csv('{}/Raw/{}.csv'.format(folder, df_name), index_col=0)
            df_processed = remove_outliers(df_raw.apply(pd.to_numeric, errors='coerce').dropna(axis=0).reset_index(drop=True), 'quantile')
            
            # The processed DataFrame is then exported to the "Processed" folder, and plotted.
            df_processed.to_csv('{}/Processed/{}_processed.csv'.format(folder, df_name))
            df_processed.plot()

            # The plot of the processed DataFrame is saved in the "Figures" folder.
            plt.savefig('{}/Figures/{}_plot.png'.format(folder, df_name))

    print(Fore.GREEN + 'Data processed successfully' + Style.RESET_ALL)

####### Sources ########
# To understand Value data type and lock method read the following link:
# https://www.kite.com/python/docs/multiprocessing.Value  
# For suffle of array, check the next link and user "mdml" answer:
# https://stackoverflow.com/questions/19597473/binary-random-array-with-a-specific-proportion-of-ones