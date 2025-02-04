
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
    #Values for the OSC Server and Emotions

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
