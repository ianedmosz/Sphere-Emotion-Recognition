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
from brainflow.data_filter import DataFilter, DetrendOperations ##WindowOperations?13
import csv

# Imports from Empatica
### SOLO IMPORTAR ESTO AL USAR LAS FUNCIONES DE EMPATICA MODULARMENTE ###
from empaticafuncs import empatica
#import socket
#import time
#import pylsl
#import numpy as np
#import pandas as pd
#from datetime import datetime
#import csv
#import matplotlib.pyplot as plt 
#import matplotlib.animation as animation
#from scipy.fft import rfft, rfftfreq
#import json
#import sys 
#from math import log

# Imports for OSC
import argparse
import random
import time
from pythonosc import udp_client
from statistics import mean, median
from math import sqrt

## CAMBIAR POR CADA ESCENARIO - DURACIÓN TOTAL DEL CODIGO
simulation_time = 60 ## 
### ESTA PARTE DEL CÓDIGO ES LA QUE SE DEBE AÑADIR AL SCRIPT FINAL SI ES QUE LO DE EMPATICA SE VA A MANEJAR MODULARMENTE ###
# Empatica functions input parameters
step = 10 # 
e4savetimes = [*range(step,simulation_time,step)]
emp_id = '834ACD'#'834ACD'  #'8839CD' #'1451CD' # 'A02088' #'A01FC2' #'de6f5a'
###FIN###

# # CODE FOR P300 TEST # #
# # You can play a sound with your speakers with the next line, uncomment if needed to run a diagnosis # #
#fc.LeftRightCheck.run()

# # Create objects that store the beep sounds using simpleaudio # #
# This sounds can include a method called .wait_done() that will literally pause everything
# until the sound has finished.

# WARNING: The sound files must be on the same directory to have the relative path, however if they are
# on different directories, you must add the ENTIRE path, example: (r"C:\user\directory\TransitionBeep.wav")
# Do NOT forget to include the "r" before the double quotes, else it will cause an error.
#transitionBeep_sound = sa.WaveObject.from_wave_file(r'Files\TransitionBeep.wav')
#transitionBeep = transitionBeep_sound.play()
#transitionBeep.wait_done()

#frequentBeep_sound = sa.WaveObject.from_wave_file(r'Files\FrequentBeep.wav')
#frequentBeep = frequentBeep_sound.play()
#frequentBeep.wait_done()

#NotFrequentBeep_sound = sa.WaveObject.from_wave_file(r'Files\NotFrequentBeep.wav')
#NotFrequentBeep = NotFrequentBeep_sound.play()
#NotFrequentBeep.wait_done()

# # Create a Value data object # #
# This object can store a single integer and share it across multiple parallel
# processes
seconds = Value("i", 0)
counts = Value("i", 0)

# # Define Parallel Processes # #
# This function is the countup timer, the count is set to 0 before the script
# waits for a second, otherwise the beep will sound several times before the second 
# changes.
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

    #s = Sender()
    #s.post_empatica_record(TemporalArray)
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
    folder = r'Newtests3\P300_S{}R{}_{}'.format(subject_ID, repetition_num, datetime.now().strftime("%d%m%Y_%H%M"))
    os.mkdir(folder)

    for subfolder in ['Raw', 'Processed', 'Figures']:
        os.mkdir('{}/{}'.format(folder, subfolder))

    # # Create a multiprocessing List # # 
    # This list will store the seconds where a beep was played
    timestamps = multiprocessing.Manager().list()
    
    # # Start processes # #
    #process1 = Process(
    #    target=beep, args=[seconds, counts, distribution_array, timestamps])
    process2 = Process(
        target=timer, args=[seconds, counts, timestamps])
    p = Process(target=empatica, args=[seconds, folder, client, client2, e4savetimes, simulation_time, emp_id]) #Descomentar para Empatica
    #q = Process(target=EEG, args=[seconds, folder])
    #m = Process(target=MachineLearning, args=[seconds, folder])
    #process1.start()
    process2.start()
    p.start() # Descomentar para Empatica
    #q.start()
    #m.start()
    #process1.join()
    process2.join()
    p.join() # Descomentar para Empatica
    #q.join()
    #m.join()

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