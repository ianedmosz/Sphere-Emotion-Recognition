import cv2
import numpy as np
import threading
import queue

class Detect:
    def __init__(self):
        self.emotion_colors = {
            "Joy": (0, 255, 255),
            "Sadness": (255, 0, 0),
            "Hate": (0, 0, 255),
            "Desire": (255, 255, 0),
            "Admiration": (239, 184, 16),
            "Love": (255, 0, 128),
        }
        self.command_queue = queue.Queue()
        self.window_thread = threading.Thread(target=self._window_loop)
        self.window_thread.daemon = True
        self.window_thread.start()

    def _window_loop(self):
        cv2.namedWindow("Emocion Detectada", cv2.WINDOW_NORMAL)
        while True:
            try:
                emotion = self.command_queue.get(timeout=0.1)
                if emotion == "EXIT":
                    break
                    
                background = np.zeros((480, 640, 3), dtype=np.uint8)
                background[:] = self.emotion_colors.get(emotion, (128, 128, 128))
                cv2.putText(background, f"Emocion: {emotion}", (50, 50),
                           cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
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
