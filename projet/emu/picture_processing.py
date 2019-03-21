import cv2
import numpy as np


class OldPictureReducer:

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


class PictureReducer:

    def __init__(self):
        self.memory = np.array([np.zeros(60 * 24 * 3) for _ in range(3)])

    @classmethod
    def get_dim(cls) -> int:
        # return (60 * 24 * 3)
        return 3

    def _reduce(self, screen: [[[]]]) -> np.ndarray:
        screen = cv2.cvtColor(screen, cv2.COLOR_RGB2BGR)
        # cv2.imshow('input', screen)
        # cv2.waitKey(1)
        screen = cv2.resize(screen, (60, 48))
        screen = screen[20:44, 0:60]
        # cv2.imshow('resized', screen)
        # cv2.waitKey(1)
        screen = screen.flatten()
        return np.reshape(screen, (60 * 24 * 3))

    def _memorize(self, screen: np.ndarray) -> [np.ndarray]:
        if screen is not self.memory[0]:
            self.memory = np.append(screen, self.memory)[0:3]
        #print(self.memory)
        #print(screen)
        return self.memory

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
        resized_screen = self._reduce(screen)
        memorized = self._memorize(resized_screen)
        # separated = []
        # for i in memorized:
        #    separated.append(self._split(i))
        return memorized
