import cv2
import time
import numpy as np

class Detect:
    def __init__(self):
        self.emotion_colors = {
            "Joy": (0, 255, 255),      # Amarillo
            "Sadness": (255, 0, 0),    # Azul
            "Hate": (0, 0, 255),       # Rojo
            "Desire": (255, 255, 0),   # Cian
            "Admiration": (239, 184, 16), # Dorado
            "Love": (255, 0, 128),     # Rosa
        }

        # Crear una ventana persistente
        cv2.namedWindow("Emocion Detectada", cv2.WINDOW_NORMAL)

    def obtain_emotions(self, emotion):
        # Obtener el color correspondiente
        background_color = self.emotion_colors.get(emotion, (128, 128, 128))  # Gris por defecto

        # Crear la imagen de fondo
        background_image = np.zeros((480, 640, 3), dtype=np.uint8)
        background_image[:] = background_color

        # Mostrar el texto de la emoción
        cv2.putText(background_image, f"Emocion: {emotion}", (50, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)

        # Mostrar la imagen
        cv2.imshow("Emocion Detectada", background_image)
        cv2.waitKey(1)  # Permite que la ventana de OpenCV responda

    def close_window(self):
        """Cierra la ventana de OpenCV."""
        cv2.destroyAllWindows()
