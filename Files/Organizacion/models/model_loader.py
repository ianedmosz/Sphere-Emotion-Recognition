import pickle
from pathlib import Path
from config.configs import variables

models_base_path=Path(__file__).parent

def load_models():
    evaluation_type = variables['test_parms'].get('evaluation_type', 'both')

    Val_Pkl_linear = Aro_Pkl_linear = Dom_Pkl_linear = Val_Pkl_cubic = Aro_Pkl_cubic = Dom_Pkl_cubic = None

    try:
        if evaluation_type == 'spherical' or evaluation_type == 'both':
            Val_Pkl_linear = pickle.load(open(models_base_path / "reg_val_model.pkl", "rb"))
            Aro_Pkl_linear = pickle.load(open(models_base_path  / "reg_aro_model.pkl", "rb"))
            Dom_Pkl_linear = pickle.load(open(models_base_path / "reg_dom_model.pkl", "rb"))

        if evaluation_type == 'cubic' or evaluation_type == 'both':
            Val_Pkl_cubic = pickle.load(open(models_base_path  / "Val_RF_10s.pkl", "rb"))
            Aro_Pkl_cubic = pickle.load(open(models_base_path  / "Aro_RF_10s.pkl", "rb"))
            Dom_Pkl_cubic = pickle.load(open(models_base_path  / "Dom_RF_10s.pkl", "rb"))

        return Val_Pkl_linear, Aro_Pkl_linear, Dom_Pkl_linear, Val_Pkl_cubic, Aro_Pkl_cubic, Dom_Pkl_cubic

    except FileNotFoundError as e:
        print(f"Error: {e}")
        exit()
    except OSError as e:
        print(f"Error al cargar los modelos: {e}")
        exit()


