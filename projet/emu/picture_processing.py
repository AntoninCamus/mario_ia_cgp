import cv2
import numpy as np


class PictureFlattener:
    @classmethod
    def get_dim(cls) -> int:
        # Dimensions de l'écran : 240 * 256 * 3
        return 240 * 256 * 3

    def process(self, screen: [[[]]]) -> []:
        return np.array(screen).flatten()


class PictureReducer:

    @classmethod
    def get_dim(cls) -> int:
        return 30 * 24 * 3

    def process(self, screen: [[[]]]) -> []:
        screen = cv2.cvtColor(screen, cv2.COLOR_RGB2BGR)
        screen = cv2.resize(screen, (30, 24))
        # cv2.imshow('main', screen)
        # cv2.waitKey(1)
        screen = screen.flatten()
        screen = np.reshape(screen, 30 * 24 * 3)
        return screen
