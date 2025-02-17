
from serial import Serial
import time

def setup_arduino():
    arduino_com=variables['test_parms']['arduino_com']
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



