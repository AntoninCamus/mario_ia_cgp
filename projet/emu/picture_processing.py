import cv2
import numpy as np


class OldPictureReducer:

    @classmethod
    def get_dim(cls) -> int:
        return 30 * 24 * 3

    def process(self, screen: [[[]]]) -> []:
        screen = cv2.cvtColor(screen, cv2.COLOR_RGB2BGR)
        screen = cv2.resize(screen, (30, 24))
        screen = screen.flatten()
        screen = np.reshape(screen, 30 * 24 * 3)
        return screen


class PictureReducer:

    def __init__(self):
        self.memory = np.array([np.zeros(60 * 24 * 3) for _ in range(3)])

    @classmethod
    def get_dim(cls) -> int:
        return (60 * 24 * 3)

    def _reduce(self, screen: [[[]]]) -> np.ndarray:
        screen = cv2.cvtColor(screen, cv2.COLOR_RGB2BGR)
        screen = cv2.resize(screen, (60, 48))
        screen = screen[20:44, 0:60]
        screen = screen.flatten()
        return np.reshape(screen, (60 * 24 * 3))

    def _split(self, screen: np.ndarray) -> [np.ndarray]:
        '''Split the target image into its red, green and blue channels.
        image - a numpy array of shape (rows, columns, 3).
        output - a numpy array containing three numpy arrays of shape (rows, columns)
                 and dtype same as image, containing the corresponding channels.
        '''

        red = screen[:, :, 2]
        green = screen[:, :, 1]
        blue = screen[:, :, 0]
        print(red)
        return [red, green, blue]

    def process(self, screen: [[[]]], render=True) -> [np.ndarray]:
        return self._reduce(screen)
