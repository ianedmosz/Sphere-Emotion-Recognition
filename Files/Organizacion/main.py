import pandas as pd
import time
from scipy import signal
import numpy as np
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
from pathlib import Path
import datetime
from models.model_loader import load_models
from config.configs import variables
from config.emotions import load_emotions
from processing.psd_bands import compute_psd_bands
from processing.butter import butter_bandpass_filter
from processing.sphere import get_emotion_sphere, get_fear_sphere
from hardware.board import setup_board
from hardware.arduino import setup_arduino
from config.emocion import emocion
from config.real_emotion import real_emotion
########################################################################################
base_path = Path(__file__).parent
#Variable a modificar
#final_descion=int(input('0 for only emotions; 1 for only fear; 2 for both:'))


# Define the function to compute PSD for multiple frequency bands
#final_descion=int(input('0 for only emotions; 1 for only fear; 2 for both:'))

emotions,emotions_fear,Descartes_Passions=load_emotions()

#emotions_imp=load_emotions()

Val_Pkl_linear, Aro_Pkl_linear, Dom_Pkl_linear, Val_Pkl_cubic, Aro_Pkl_cubic, Dom_Pkl_cubic = load_models()

iteraciones = 0
warnings.filterwarnings("ignore", category=UserWarning, module="sklearn")
warnings.filterwarnings("ignore", category=DataConversionWarning)

ip=variables['ip'] #Direccion IP
ports=variables['ports'] #Puertos Definidos
address=variables['address'] #Direcciones OSC


port1 = ports['port1']['value'] 
port2 = ports['port2']['value']
port5 = ports['port5']['value'] #arduino
port3 = ports['port3']['value'] # TD miedo
port4 = ports['port4']['value'] #PD miedo

client1 = udp_client.SimpleUDPClient(ip, port1) #HTTP
client2 = udp_client.SimpleUDPClient(ip, port2)#API

client3 = udp_client.SimpleUDPClient(ip, port3)#Base de datos(TD Miedo)
client4 = udp_client.SimpleUDPClient(ip, port4)#Datos Adicionales(PD Miedo)
client5 = udp_client.SimpleUDPClient(ip, port5) #arduino

address_engagement = address['address']  # /engagement
address_emotion = address['address2']  # /real-emotion
address_emotion_id = address['address3']  # /emotion-ID
address_fear = address['address4']  # /Fear-emotion
address_similarity = address['address5']  # /emotion-similarity
#Testparams

arduino_bol=variables['test_parms']['arduino_bol']
synthetic_bol=variables['test_parms']['synthetic_bol']
linear_bol=variables['test_parms']['linear_bol']
arduino_com=variables['test_parms']['arduino_com']
open_bci_com=variables['test_parms']['open_bci_com']
final_descion=variables['test_parms']['final_descion']

# Set the duration to stream data (5 seconds in this example)
duration = variables['test_parms']["duration"] # queremos 10 segundos
# Set the sampling rate and channel(s) you want to stream

## SEPTEMBER 28 UPDATE ##
sampling_rate =variables['test_parms']['sampling_rate'] # used to be 128, queremos 250 Hz

#channels = (0, 1, 2, 3, 4, 5, 6, 7)  # Streaming data from channels 0 to 7
channels = list(range(9))   # Streaming data from channels 0 to 7\n",
#Path para dicionarios de emociones

evaluation_type = variables['test_parms'].get('evaluation_type', 'both')


###############################################################################




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
lodb = f"{base_path}/Data"
folder = f'S{subject_ID}R{repetition_num}_{datetime.datetime.now().strftime("%d%m%Y_%H%M")}'
os.mkdir(f"{lodb}/{folder}")

df_eeg = pd.DataFrame()
df_time = pd.DataFrame()
df_emotions = pd.DataFrame(data = [], columns=['Timestamp', 'Engagement', 'Emotion', 'Valence', 'Arousal', 'Dominance'])
df_acc = pd.DataFrame() # acceleration
df_fear=pd.DataFrame(data = [], columns=['Timestamp', 'Engagement', 'Emotion', 'Valence', 'Arousal', 'Dominance'])
df_spherical = pd.DataFrame(columns=['Timestamp', 'Engagement', 'Spherical_Emotion', 'Temp', 'Sphere_Vector', 'Valence', 'Arousal', 'Dominance'])
df_cubic = pd.DataFrame(columns=['Timestamp', 'Engagement', 'Cubic_Emotion', 'Valence', 'Arousal', 'Dominance'])
df_fear_spherical = pd.DataFrame(columns=['Timestamp', 'Engagement', 'Fear_Metric', 'Theta'])
print(Fore.RED + 'Initializing functions...' + Style.RESET_ALL)


def classify_fearm_metric(fear_metric):
    bins= [0, 0.25, 0.50, 0.75, 1.0]
    labels = ["Null", "Low", "Medium", "High"]
    index = np.digitize(fear_metric, bins, right=True) - 1
    return labels[index]

spherical_emotion = "Unknown"

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
            df_time = df_time.append(np_df)
            
            ## ACCELERATION ##
            
            for i, channel in enumerate(acc_channel):
                channel_data_acc[i].extend(samples[channel])
                                

            # Sleep for a small interval to avoid high CPU usage
            time.sleep(1)

        # Stop the streaming
        board.stop_stream()

        # Stop the streaming
        
        # acceleration dataframea
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
        
        vale, arou, domin, domi = 0, 0, 0, 0
        iteraciones += 1

        if evaluation_type == 'cubic':
            valen_cubic = Val_Pkl_cubic.predict(df_pred)
            arous_cubic = Aro_Pkl_cubic.predict(df_pred)
            domin_cubic = Dom_Pkl_cubic.predict(df_pred)
            vale_cubic = mean(valen_cubic) 
            arou_cubic = mean(arous_cubic) 
            domi_cubic = mean(domin_cubic) 
            vale, arou, domi = vale_cubic, arou_cubic, domi_cubic
            

        elif evaluation_type == 'spherical':
            valen_linear = Val_Pkl_linear.predict(df_pred)
            arous_linear = Aro_Pkl_linear.predict(df_pred)
            domin_linear = Dom_Pkl_linear.predict(df_pred)
            vale_linear = mean(valen_linear) 
            arou_linear = mean(arous_linear) 
            domi_linear = mean(domin_linear) 
            print(f"V{vale_linear:.2f}, A{arou_linear:.2f}, D{domi_linear:.2f}")
            vale, arou, domi = vale_linear, arou_linear, domi_linear
            

           # print(f"V{vale:.2f},A{arou:.2f},D{domi:.2f}")

        elif evaluation_type == 'both':
            # Predicciones cúbicas
            valen_cubic = Val_Pkl_cubic.predict(df_pred)
            arous_cubic = Aro_Pkl_cubic.predict(df_pred)
            domin_cubic = Dom_Pkl_cubic.predict(df_pred)
            vale_cubic = mean(valen_cubic) 
            arou_cubic = mean(arous_cubic) 
            domi_cubic = mean(domin_cubic) 

             # Predicciones lineales
            valen_linear = Val_Pkl_linear.predict(df_pred)
            arous_linear = Aro_Pkl_linear.predict(df_pred)
            domin_linear = Dom_Pkl_linear.predict(df_pred)
            vale_linear = mean(valen_linear) 
            arou_linear = mean(arous_linear) 
            domi_linear = mean(domin_linear) 

            vale, arou, domi = vale_linear, arou_linear, domi_linear
            
                 
        
        engag_fp1 = mean(df_pred["Fp1_Engagement"])
        engag_fp2 = mean(df_pred["Fp2_Engagement"])
        engag_c3 = mean(df_pred["C3_Engagement"])
        engag_c4 = mean(df_pred["C4_Engagement"])
        engag_p7 = mean(df_pred["P7_Engagement"])
        engag_p8 = mean(df_pred["P8_Engagement"])
        engag_o1 = mean(df_pred["O1_Engagement"])
        engag_o2 = mean(df_pred["O2_Engagement"])

        if evaluation_type in ['cubic', 'both']:
            print(f"Cubic V{vale_cubic:.2f}, A{arou_cubic:.2f}, D{domi_cubic:.2f}")
        if evaluation_type in ['linear', 'both']:
            print(f"Linear V{vale_linear:.2f}, A{arou_linear:.2f}, D{domi_linear:.2f}")


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
        
        #Definicion de cuales funciones se mandan a llamar en base al modelo linear o cubico
        
        if evaluation_type in ['cubic', 'both']:
            cubic_emotion = emocion(final_descion,vale, arou, domi, emotions, emotions_fear)
            print(f"Cubic V{vale_cubic:.2f}, A{arou_cubic:.2f}, D{domi_cubic:.2f}")

            if final_descion == 0:  # Solo emociones
                print(f"Cubic Emotion: {cubic_emotion}")
            elif final_descion == 1:  # Solo miedo
                print(f"Cubic Fear: {cubic_emotion.replace('Fear', '').strip()}")
            elif final_descion == 2:  # Ambos
                cubic_emotion = emocion(final_descion,vale_cubic, arou_cubic, domi_cubic, emotions, emotions_fear)
                emotion_value, fear_value = cubic_emotion
                print(f"Cubic Emotion: {emotion_value}, Fear: {fear_value.replace('Fear', '').strip()}")

            if final_descion in [0, 2]:  # Guardar en el DataFrame
                df_cubic.loc[len(df_cubic)] = [
                    time.time() - 21600,  # Timestamp ajustado a GMT-6
                    engagement,           # Engagement calculado
                    cubic_emotion if final_descion == 0 else emotion_value,  # Emoción cúbica o valor emocional
                    vale_cubic, arou_cubic, domi_cubic  # Valores de valencia, arousal y dominancia
                ]

        if evaluation_type in ['spherical', 'both']:
            detected_values = np.array([vale, domi, arou]).flatten()  # Valores normalizados para la esfera emocional
            temp, fear_metric, theta = 0, 0, 0
            spherical_emotion, sphere_vector = "Unknown", None
            print(f"Linear V{vale:.2f}, A{arou:.2f}, D{domi:.2f}")


            # Predicción de emociones esféricas
            if final_descion in [0, 2]:
                try:
                    spherical_emotion, temp, sphere_vector = get_emotion_sphere(detected_values)
                except Exception as e:
                    print(f"Error en la predicción esférica: {e}")

            # Predicción del miedo en la esfera
            if final_descion in [1, 2]:
                try:
                    fear_metric, theta = get_fear_sphere(detected_values)
                    fear_label = classify_fearm_metric(fear_metric)
                except Exception as e:
                    print(f"Error en la predicción de miedo: {e}")
                    fear_metric, theta = None, None


            # Guardar métricas de miedo en el DataFrame
                df_fear_spherical.loc[len(df_fear_spherical)] = [
                    time.time() - 21600,  # Timestamp ajustado a GMT-6
                    engagement,           # Engagement calculado
                    fear_metric,          # Métrica de miedo
                    theta                 # Ángulo theta
                ]

            # Mostrar resultados de predicciones
            if final_descion == 0:
                print(f"Spher Emotion: {spherical_emotion} {temp * 100:.2f}%")
            elif final_descion == 1:
                print(f"Spher Fear: {fear_label} {fear_metric * 100:.2f}%")
            elif final_descion == 2:
                print(f"Spher Emotion: {spherical_emotion} {temp * 100:.2f}%, Fear: {fear_label} {fear_metric * 100:.2f}%")

         # Guardar emociones esféricas en el DataFrame
            if final_descion in [0, 2]:
                df_spherical.loc[len(df_spherical)] = [
                    time.time() - 21600,  # Timestamp ajustado a GMT-6
                    engagement,           # Engagement calculado
                    spherical_emotion,    # Emoción esférica
                    temp,                 # Similitud con la emoción
                    sphere_vector.tolist() if sphere_vector is not None else None, # Vector esférico normalizado
                    vale, arou, domi  
                ]


        emociones = emocion(final_descion,arou, domi, vale,emotions,emotions_fear) #Aqui no se 
        realemotion = real_emotion(spherical_emotion)  # ID categórico de la emoción

        def send_scaled_metric(client, address, metric):
            scaled_metric = round(metric * 3 / 100, 2)
            client.send_message(address, scaled_metric)


        #write_read(vale, arou, domi)
        values = [vale, arou, domin]
        realemotion = real_emotion(emociones)

        # Engagement (nivel de compromiso)
        client1.send_message(address_engagement, engag)
        client2.send_message(address_engagement, engag)

        client3.send_message(address_emotion, spherical_emotion)
        client3.send_message(address_emotion_id, realemotion)

        client4.send_message(address_emotion, spherical_emotion)
        client4.send_message(address_emotion_id, realemotion)

        client5.send_message(address_emotion, spherical_emotion)
        client5.send_message(address_emotion_id, realemotion)

        if evaluation_type in ['spherical', 'both']:
            send_scaled_metric(client3, address_similarity, temp)
        if final_descion in [1, 2]:
            send_scaled_metric(client3, address_similarity, fear_metric)
        
        
        if final_descion == 0:
            df_emotions.loc[len(df_emotions)] = [time.time() - 21600, engag, emociones, vale, arou, domi]
        elif final_descion == 1:
            df_fear.loc[len(df_fear)] = [time.time() - 21600, engag, emociones, vale, arou, domi]
        elif final_descion == 2:
            df_emotions.loc[len(df_emotions)] = [time.time() - 21600, engag, emociones[0], vale, arou, domi]
            df_fear.loc[len(df_fear)] = [time.time() - 21600, engag, emociones[1], vale, arou, domi]

        df_emotions.style.format("{:.2f}")   
        #print(emociones) #descomentar
        print(f"Engagement:{engag*100:.2f}%") #descomentar 
        df_fear.style.format("{:.2f}")  
        print("")
        # print(fear_cal)

    board.stop_stream()

except KeyboardInterrupt:
    board.stop_stream()
    print(Fore.BLUE + 'Test interrupted. Storing data...' + Style.RESET_ALL)

    # Resetear y renombrar DataFrames
    df_eeg = df_eeg.reset_index(drop=True)
    df_time = df_time.reset_index(drop=True)
    df_time.columns = ['Timestamp']
    df_emotions = df_emotions.reset_index(drop=True)
    df_fear = df_fear.reset_index(drop=True)
    df_acc = df_acc.reset_index(drop=True)
    df_acc.columns = ['Acc_1', 'Acc_2', 'Acc_3']

    # Concatenar datos completos (EEG + Tiempos + Aceleración)
    df_complete = pd.concat([df_time, df_eeg, df_acc], axis=1)
    df_complete = df_complete.reset_index(drop=True)

    # Definir rutas para los archivos
    complete_path = f"{folder}/Complete_Data.csv"
    cubic_path = f"{folder}/Cubic_Emotions.csv"
    spherical_path = f"{folder}/Spherical_Emotions.csv"
    fear_spherical_path = f"{folder}/Fear_Spherical.csv"
    styled_table_path = f"{folder}/styled_table.html"

    # Guardar los datos completos
    df_complete.to_csv(complete_path, index=False)

    if not df_cubic.empty:
        df_cubic.to_csv(cubic_path, index=False)

    if not df_spherical.empty:
        df_spherical.to_csv(spherical_path, index=False)
    
    if not df_fear_spherical.empty:
        df_fear_spherical.to_csv(fear_spherical_path, index=False)

  
    def load_table(file_path, tail_rows=20):
        if os.path.exists(file_path):
            print(f"Loading file: {file_path}")
            return pd.read_csv(file_path).tail(tail_rows)
        else:
            print(f"Error: File '{file_path}' not found.")
            return pd.DataFrame()

    combined_table = pd.DataFrame()

        # Evaluación cúbica
    if evaluation_type in ['cubic', 'both'] and os.path.exists(cubic_path):
        table_cubic = load_table(cubic_path)
        if final_descion == 0:
            combined_table = table_cubic
        elif final_descion == 2:
            combined_table = pd.concat([combined_table, table_cubic], ignore_index=True)

        # Evaluación esférica
    if evaluation_type in ['spherical', 'both'] and os.path.exists(spherical_path):
        table_spherical = load_table(spherical_path)
        if final_descion == 1:
            combined_table = table_spherical
        elif final_descion == 2:
            combined_table = pd.concat([combined_table, table_spherical], ignore_index=True)

    if evaluation_type in ['spherical', 'both'] and os.path.exists(fear_spherical_path):
        table_fear = load_table(fear_spherical_path)
        if final_descion == 1:
            combined_table = table_fear
        elif final_descion == 2:
            combined_table = pd.concat([combined_table, table_fear], ignore_index=True)

        # Mostrar últimas filas de las combinaciones
    if not combined_table.empty:
        print(f"Last rows of combined data:\n{combined_table.tail(20)}")

        # Estilizar y guardar tabla como HTML
        num_columns = ['Timestamp', 'Engagement', 'Valence', 'Arousal', 'Dominance']
        styled_table = (
            combined_table.style
            .format({col: "{:.2f}" for col in num_columns if col in combined_table.columns})
            .highlight_max(subset=['Engagement'], axis=0, color='pink')
            .highlight_min(subset=['Engagement'], axis=0, color='blue')
        )

            # Guardar la tabla estilizada
        with open(styled_table_path, 'w') as f:
            f.write(styled_table.render())
            print(f"Styled table saved to {styled_table_path}")
    else:
        print("No combined data available to display or save.")

        # Mostrar datos almacenados
        print(f'Emotions:\n{df_emotions}')
        print(f'Fear:\n{df_fear}')

    board.stop_stream()
    board.release_session()

except Exception as e:
    print(Fore.RED + f"Error while processing data: {e}" + Style.RESET_ALL)

finally:
    # Detener la sesión de la placa
    print(Fore.GREEN + 'Data stored successfully.' + Style.RESET_ALL)


