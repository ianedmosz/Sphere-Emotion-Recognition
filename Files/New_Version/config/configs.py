import json 
from pathlib import Path

base_path=Path(__file__).parent

def load_config():
    config_path=base_path/"Variables.json"
    try:
        with open(config_path,"r") as f:
            return json.load(f)
    except FileNotFoundError:
        print(base_path)
        print("Error: File not found. config")
        exit()
    except json.JSONDecodeError:
        print(base_path)
        print("The file has an invalid format. ")
        exit()

variables=load_config()

