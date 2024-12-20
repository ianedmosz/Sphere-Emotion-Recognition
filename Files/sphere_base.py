import numpy as np

Descartes_Passions = {
    'Desire': np.array([1,0,0]),
    'Admiration': np.array([0,0,1]),
    'Joy': np.array([1,0,1]),
    'Love': np.array([1,-1,1]),
    'Hate': np.array([-1,-1,1]),
    'Sadness': np.array([-1, 0, -1])
}

DP_NORM = {emotion: vec / np.linalg.norm(vec) for emotion, vec in Descartes_Passions.items()}

def map_value(val):
    return (val - 1) / (4) - 1

def get_emotion_sphere(value):
    """
    Identify the closest matching emotion vector to the given input.

    Input:
    - value (np.array): The input vector representing a detected value for emotion analysis.
        where
        value[0]: valence [1,9], 
        value[1]: dominance [1,9],
        value[2]: arousal [1,9]

    Returns:
    - final_emotion (str): The emotion label with the highest similarity to the normalized input.
    - temp (float): The highest similarity score found.
    - random_emotion (np.array): The normalized version of the input vector on the unit sphere.
    """

    # Map or adjust the input value to [-1,1].
    detected_value = map_value(value)

    # Normalize the mapped detected_value vector to unit length to ensure it lies on the unit sphere.
    random_emotion = detected_value / np.linalg.norm(detected_value)

    # Initialize a temporary variable to store the highest similarity score found.
    temp = 0
    
    # Iterate over each emotion and its normalized vector in DP_NORM.
    for emotion, vec in DP_NORM.items():
        
        # Calculate the cosine similarity between random_emotion and the current emotion vector.
        # Adding 1 and dividing by 2 scales the similarity from [-1, 1] to [0, 1].
        dot = (np.dot(random_emotion, vec) + 1) / 2
        
        # Update temp and final_emotion if the current similarity score is the highest encountered.
        if dot > temp:
            temp, final_emotion = dot, emotion

    # Return the final emotion label with the highest similarity score, the score itself,
    # and the normalized random_emotion vector.
    return final_emotion, temp, random_emotion

import numpy as np

def get_rot_matrix():
    """
    Generates a 3x3 rotation matrix based on predefined angles theta and phi.
    
    The rotation matrix is constructed as:
        - Rows correspond to transformations in 3D space (x, y, z).
        - Uses trigonometric functions to compute rotational transformations.

    Returns:
        np.ndarray: A 3x3 rotation matrix.
    """
    # Define angles for rotation in radians
    theta = 2.356194490192345  # Rotation angle around the z-axis
    phi = -0.9553166181245093  # Rotation angle around the x-axis

    # Construct the rotation matrix using the angles
    return np.array([
        [np.cos(theta) * np.cos(phi), -np.sin(theta) * np.cos(phi), np.sin(phi)],
        [np.sin(theta), np.cos(theta), 0],
        [-np.sin(phi) * np.cos(theta), np.sin(phi) * np.sin(theta), np.cos(phi)]
    ])

def get_fear_sphere(value):
    """
    Maps a value to a rotated 3D vector and computes spherical coordinates.

    The function maps an input value to a 3D space using the rotation matrix,
    calculates spherical coordinates (phi and theta), and determines a fear
    metric based on the angle difference from the zenith.

    Args:
        value (np.array): The input vector representing a detected value for emotion analysis.
        where
            value[0]: valence [1,9], 
            value[1]: dominance [1,9],
            value[2]: arousal [1,9]

    Returns:
        tuple: A pair containing:
            - The fear metric (distance from pi normalized by pi).
            - The angle theta (azimuthal angle in the xy-plane).
    """
    # Get the rotation matrix
    r = get_rot_matrix()

    # Apply the rotation matrix to the mapped vector
    # Map or adjust the input value to [-1,1].
    n_vec = np.dot(r, map_value(value))
    
    # Decompose the rotated vector into components
    valence, dominance, arousal = n_vec
    
    # Compute the polar angle phi (angle from the z-axis)
    phi = np.arctan2(np.sqrt(valence ** 2 + dominance ** 2), arousal)
    
    # Compute the azimuthal angle theta (angle in the xy-plane)
    theta = np.arctan2(valence, dominance)

    # Calculate the fear metric as the normalized absolute deviation of phi from pi
    return np.abs(phi - np.pi) / np.pi, theta

def get_fear_sphere_GA(value):
    """
    Maps a value to a rotated 3D vector and computes spherical coordinates.

    The function maps an input value to a 3D space using the rotation matrix,
    calculates spherical coordinates (phi and theta), and determines a fear
    metric based on the angle difference from the zenith.

    Args:
        - value (np.array): The input vector representing a detected value for emotion analysis.
        where
            value[0]: valence [1,9], 
            value[1]: dominance [1,9],
            value[2]: arousal [1,9]

    Returns:
        tuple: A pair containing:
            - The fear metric (clipped from 0 to 1).
    """

    # Get the rotation matrix
    r = get_rot_matrix()

    # Apply the rotation matrix to the mapped vector
    # Map or adjust the input value to [-1,1].
    n_vec = np.dot(r, map_value(value))
    
    valence, dominance, _ = value
    # Decompose the rotated vector into components
    valence_rot, dominance_rot, arousal_rot = n_vec
    
    # Compute the polar angle phi (angle from the z-axis)
    phi = np.arctan2(np.sqrt(valence_rot ** 2 + dominance_rot ** 2), arousal_rot)
    
    # Compute the azimuthal angle theta (angle in the xy-plane)
    theta = np.arctan2(valence, dominance)

    # Calculate the fear metric clipped from 0 to 1
    return np.clip(phi + np.cos(theta + np.pi/4) * 0.2792519508607794, 0, 1)