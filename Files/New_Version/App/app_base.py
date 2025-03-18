import cv2
import time
import numpy as np

# Definir colores de fondo para cada emoción (en formato BGR)
emotion_colors = {
    "happy": (0, 255, 255),  # Amarillo
    "sad": (255, 0, 0),      # Azul
    "angry": (0, 0, 255),    # Rojo
    "surprise": (255, 255, 0), # Cian
    "neutral": (128, 128, 128), # Gris
    "admiration": (239, 184, 16), # Dorado
    "love": (255, 0, 128),    # Rosa
}

# Lista de emociones a simular
emotions_to_simulate = ["happy", "sad", "angry", "surprise", "neutral"]

# Crear una ventana para mostrar la emoción detectada
cv2.namedWindow("Emocion Detectada", cv2.WINDOW_NORMAL)

# Simular cambios de emoción
for emotion in emotions_to_simulate:
    # Obtener el color de fondo correspondiente a la emoción actual
    background_color = emotion_colors.get(emotion, (128, 128, 128))  # Gris por defecto

    # Crear una imagen sólida con el color de fondo
    background_image = np.zeros((480, 640, 3), dtype=np.uint8)  # Imagen negra de 640x480
    background_image[:] = background_color  # Rellenar con el color de la emoción

    # Mostrar la emoción detectada como texto en la imagen
    cv2.putText(background_image, f"Emocion: {emotion}", (50, 50),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)

    # Mostrar la imagen con el fondo de color
    cv2.imshow("Emocion Detectada", background_image)

    # Esperar 2 segundos antes de cambiar a la siguiente emoción
    time.sleep(2)

    # Salir del bucle si se presiona la tecla 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Cerrar la ventana al finalizar
cv2.destroyAllWindows()