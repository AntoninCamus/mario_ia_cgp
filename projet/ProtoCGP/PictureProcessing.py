class PictureFlattener:
    @classmethod
    def get_dim(cls) -> int:
        # Dimensions de l'écran : 240 * 256 * 3
        return 240 * 256 * 3

    def process(self, screen: [[[]]]) -> []:
        res = []
        for column_of_pixels in screen:
            res += [color for pixel in column_of_pixels for color in pixel]
        return res


class PictureBasicAnalyser:
    @classmethod
    def get_dim(cls) -> int:
        NotImplemented

    def process(self, screen: [[[]]]) -> []:
        NotImplemented
