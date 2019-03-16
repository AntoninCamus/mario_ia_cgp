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
        return 60 * 56

    def process(self, screen: [[[]]]) -> []:
        screen = cv2.cvtColor(screen, cv2.COLOR_BGR2GRAY)
        screen = cv2.resize(screen, (60, 56))
        cv2.imshow('main', screen)
        cv2.waitKey(1)
        screen = np.reshape(screen, (60, 56))
        return screen.flatten()

        # 1 PIX = 16*16


class PictureAnalyser:
    # {(104, 136, 252): 0, (252, 252, 252): 1, (252, 160, 68): 2, (228, 92, 16): 3,
    # (0, 0, 0): 4, (184, 248, 24): 5, (240, 208, 176): 6, (248, 56, 0): 7}

    @classmethod
    def get_dim(cls) -> int:
        return 15 * 14

    def process(self, screen: [[[]]]) -> []:
        return NotImplemented
