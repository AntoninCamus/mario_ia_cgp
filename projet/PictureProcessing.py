import numpy as np


class PictureFlattener:
    @classmethod
    def get_dim(cls) -> int:
        # Dimensions de l'écran : 240 * 256 * 3
        return 240 * 256 * 3

    def process(self, screen: [[[]]]) -> []:
        # res = []
        # for column_of_pixels in screen:
        #    res += [color for pixel in column_of_pixels for color in pixel]
        # return res
        return np.array(screen).flatten()


class PictureBasicAnalyser:
    map: {(int, int, int): int}

    @classmethod
    def get_dim(cls) -> int:
        NotImplemented

    def process(self, screen: [[[]]]) -> []:
        CompressedPicture = [[[]]]
        # 1 PIX = 16*16
        # {(104, 136, 252): 0, (252, 252, 252): 1, (252, 160, 68): 2, (228, 92, 16): 3, (0, 0, 0): 4, (184, 248, 24): 5, (240, 208, 176): 6, (248, 56, 0): 7}
        NotImplemented
