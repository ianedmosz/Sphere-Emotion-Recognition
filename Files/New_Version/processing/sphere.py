# emotion_utils.py
import numpy as np
from config.emotions import Descartes_Passions 

# Suponiendo que 'Descartes_Passions' es un diccionario global
DP_NORM = {emotion: vec / np.linalg.norm(vec) for emotion, vec in Descartes_Passions.items()}

# Función para mapear valores
def map_value(val):
    return (2) * (val - 1) / (8) - 1

# Función para obtener la emoción más cercana en la esfera
def get_emotion_sphere(value):
    """
    Identify the closest matching emotion vector to the given input.
    
    Input:
    - value (np.array): The input vector representing a detected value for emotion analysis.
    
    Returns:
    - final_emotion (str): The emotion label with the highest similarity.
    - temp (float): The highest similarity score found.
    - random_emotion (np.array): The normalized version of the input vector on the unit sphere.
    """
    detected_value = map_value(value)
    random_emotion = detected_value / np.linalg.norm(detected_value)
    temp = 0
    
    for emotion, vec in DP_NORM.items():
        dot = (np.dot(random_emotion, vec) + 1) / 2
        if dot > temp:
            temp, final_emotion = dot, emotion

    return final_emotion, temp, random_emotion

# Función para obtener el "fear sphere"
def get_fear_sphere(value):
    """
    Computes the fear metric based on the given input value.
    
    Input:
    - value (np.array): The input vector representing valence, dominance, and arousal.
    
    Returns:
    - fear_metric (tuple): The computed fear metric values (angle, azimuthal angle).
    """
    theta = 2.356194490192345  # Rotation angle around the z-axis
    phi = -0.9553166181245093  # Rotation angle around the x-axis
    
    r = np.array([
        [np.cos(theta) * np.cos(phi), -np.sin(theta) * np.cos(phi), np.sin(phi)],
        [np.sin(theta), np.cos(theta), 0],
        [-np.sin(phi) * np.cos(theta), np.sin(phi) * np.sin(theta), np.cos(phi)]
    ])

    n_vec = np.dot(r, map_value(value))
    
    valence, dominance, arousal = n_vec
    phi = np.arctan2(np.sqrt(valence ** 2 + dominance ** 2), arousal)
    theta = np.arctan2(valence, dominance)

    fear_metric = np.abs(phi - np.pi) / np.pi, theta

    return fear_metric
