import numpy as np
import pyeeg as pe

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
