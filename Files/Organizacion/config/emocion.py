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
