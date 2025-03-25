import cv2
import numpy as np
import threading
import queue
from datetime import datetime

class Detect:
    def __init__(self):
        self.emotion_colors = {
            "Joy": (0, 255, 255),      # Amarillo (BGR)
            "Sadness": (255, 0, 0),    # Azul
            "Hate": (0, 0, 255),       # Rojo
            "Desire": (255, 255, 0),   # Cian
            "Admiration": (239, 184, 16), # Dorado
            "Love": (255, 0, 128),     # Rosa
            "Unknown": (128, 128, 128) # Gris por defecto
        }
        
        self.command_queue = queue.Queue()
        self.window_thread = threading.Thread(target=self._window_loop)
        self.window_thread.daemon = True
        self.window_thread.start()

    def _window_loop(self):
        cv2.namedWindow("Emocion Detectada", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("Emocion Detectada", 640, 480)  
        
        while True:
            try:
                emotion = self.command_queue.get(timeout=0.1)
                if emotion == "EXIT":
                    break
                    
                # Crear la imagen de fondo
                background = np.zeros((480, 640, 3), dtype=np.uint8)
                background[:] = self.emotion_colors.get(emotion, self.emotion_colors["Unknown"])
                
                font = cv2.FONT_HERSHEY_DUPLEX  # Fuente
                scale = 1.3 
                thickness = 2
                color = (255, 255, 255)  
                
                if emotion in ["Joy", "Desire"]:
                    color = (0, 0, 0)  # Texto negro para amarillo/cian
                
                text = f"Emotion: {emotion}"
                text_size = cv2.getTextSize(text, font, scale, thickness)[0]
                text_x = (background.shape[1] - text_size[0]) // 2
                text_y = (background.shape[0] + text_size[1]) // 2
                
                cv2.putText(background, text, (text_x+2, text_y+2), 
                           font, scale, (0, 0, 0), thickness+1, cv2.LINE_AA)
                
                cv2.putText(background, text, (text_x, text_y), 
                           font, scale, color, thickness, cv2.LINE_AA)
                
                # Mostrar la imagen
                cv2.imshow("Emocion Detectada", background)
                
            except queue.Empty:
                pass
            
            cv2.waitKey(100)

        cv2.destroyAllWindows()

    def obtain_emotions(self, emotion):
        self.command_queue.put(emotion)

    def close_window(self):
        self.command_queue.put("EXIT")
        self.window_thread.join()
