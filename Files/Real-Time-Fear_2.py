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


address = "/engagement"  # OSC address to send the message to
address2 = "/real-emotion"  # OSC address to send the message to
address3 = "/emotion-ID"
address4= "/Fear-emotions"

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

# Preguntar si el usuario quiere usar Arduino
arduino_1 = input("Do you want to use Arduino? (y/n): ").lower()
arduino_bol = arduino_1 == 'y'

# Preguntar si el usuario quiere usar valores sintéticos
synthetic_bol = input("Do you want to use synthetic values? (yes/no): ").lower() == 'y'


###############################################################################

def compute_psd_bands(data, fs):

    # Define the frequency ranges for each band
    bands = {
        'Delta': (0.5, 4),
        'Theta': (4, 8),
        'Alpha': (8, 12),
        'Beta': (12, 30),
        'Gamma': (30, 45)
    }

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
                    "Generosity": None
                    }
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
    timestamp_channel = board.get_timestamp_channel(BoardIds.CYTON_BOARD.value if not is_synthetic else BoardIds.SYNTHETIC_BOARD.value)
    acc_channel = board.get_accel_channels(BoardIds.CYTON_BOARD.value if not is_synthetic else BoardIds.SYNTHETIC_BOARD.value)

    return board, timestamp_channel, acc_channel



# 1. Sí Arduino, Sí Sintético
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

        # Create an empty DataFrame to store the PSD results
        psd_df = pd.DataFrame()

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
         {'selector': 'td', 'props': [('border', '1px solid black')]}]
    )
)

with open(f'{lodb}/{folder}/styled_table.html', 'w') as f:
    f.write(styled_table.render())

print("Table saved as 'styled_table.html'")
