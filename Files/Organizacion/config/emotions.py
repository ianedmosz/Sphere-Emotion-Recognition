from pathlib import Path
import json

base_path= Path(__file__).parent

def load_emotions():
    emotions_path=base_path/"Emotions"/"emociones.json"
    fear_path=base_path/"Emotions"/"fear.json"
    Descartes_Passions_path=base_path/"Emotions"/"descartes.json"

    try:
        with open(emotions_path,"r") as f:
            emotions=json.load(f)
        with open(fear_path,"r") as r:
            emotions_fear=json.load(r)
        with open(Descartes_Passions_path,"r") as d:
            Descartes_Passions=json.load(d)
    
    except FileNotFoundError:
        print(base_path)
        print("Error: File Not Found emotions")
        exit()

    return emotions, emotions_fear, Descartes_Passions

emotions,emotions_fear,Descartes_Passions=load_emotions()
