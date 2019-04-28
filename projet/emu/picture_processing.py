import cv2
import numpy as np


def display_rgb(screen: np.ndarray):
    cv2.imshow("main", cv2.cvtColor(screen, cv2.COLOR_RGB2BGR))
    cv2.waitKey(1)


def get_color_bitmask(color: [], image: np.ndarray) -> np.ndarray:
    return cv2.inRange(image, color, color)


class PictureReducer:
    @classmethod
    def get_dim(cls) -> int:
        return 7

    def _detect_color(
            self, screen: np.ndarray, colors_target: [np.ndarray]
    ) -> np.ndarray:
        """
        Function that will return a bitmap containing 1 iff :
        - All colors were found in the window containing this pixel
        - And if the precise pixel is of one of these colors

        :param screen: A np.ndarray of shape (x,y,3) containing at the bottom level RGB values of one pixel
        :param colors_target: A np.ndarray of shape (3) containing RGB values of colors which are aimed to find
        :return: A np.ndarray of shape (x,y) containing 1 or 0 (cf function description).
        """
        STEP_Y = 15
        STEP_X = 16

        result = np.full((screen.shape[0], screen.shape[1]), False)

        for y in range(0, len(screen), STEP_Y):
            for x in range(0, len(screen[0]), STEP_X):
                sub_screen = screen[y: y + STEP_Y, x: x + STEP_X]

                # Are all mandatory colors here ?
                col_bitmask = []
                all_found = True
                for color in colors_target:
                    bitmask = get_color_bitmask(color, sub_screen)
                    if any(bitmask.flatten()):
                        col_bitmask.append(bitmask)
                    else:
                        # One color was not found
                        all_found = False
                        break

                # If yes, we merge corresponding pixels on the bitmask :
                if all_found:
                    for b in col_bitmask:
                        result[y: y + STEP_Y, x: x + STEP_X] = np.bitwise_or(
                            result[y: y + STEP_Y, x: x + STEP_X], b
                        )
        return result

    def _reduce(self, screen: [[[]]]) -> np.ndarray:
        screen = cv2.resize(screen[100:220], (120, 80))
        # display_rgb(screen)
        return screen

    def _detect_mario(self, screen: np.ndarray) -> np.ndarray:
        """
        Detect mario in the target image.

        :param screen: a numpy array of shape (rows, columns, 3) containing subarray of [red,green,blue].
        :return: a np-array of shape of shape (rows, columns) containing a bitmap of presence of mario or not.
        """
        MARIO_RGB = np.array([[248, 56, 0], [172, 124, 0], [252, 160, 68]])
        return self._detect_color(screen, MARIO_RGB) * 1

    def _detect_monsters(self, screen: np.ndarray) -> np.ndarray:
        """
        Detect monsters in the target image.

        :param screen: a numpy array of shape (rows, columns, 3) containing subarray of [red,green,blue].
        :return: a np-array of shape of shape (rows, columns) containing a bitmap of presence of monsters or not.
        """
        GUMPA_RGB = np.array([(228, 92, 16), (240, 208, 176), (0, 0, 0)])
        TURTLE_OR_PLANT_RGB = np.array([(252, 160, 68), (0, 168, 0)])

        return (
                np.bitwise_or(
                    self._detect_color(screen, GUMPA_RGB),
                    self._detect_color(screen, TURTLE_OR_PLANT_RGB),
                )
                * 1
        )

    def _detect_shell_or_pipe(self, screen: np.ndarray) -> np.ndarray:
        TURTLESHELL_OR_PIPE_RGB = np.array([0, 168, 0])

        return (
                np.bitwise_and(
                    get_color_bitmask(TURTLESHELL_OR_PIPE_RGB, screen),
                    np.bitwise_not(get_color_bitmask(np.array([252, 160, 68]), screen)),
                )
                * 1
        )

    def _detect_floor(self, screen: np.ndarray) -> np.ndarray:
        FLOOR_COLOR = np.array([228, 92, 16])
        return get_color_bitmask(FLOOR_COLOR, screen) * 1

    def _flatten_output(self, array: [np.ndarray]):
        return [array.flatten() for array in array]

    def process(self, screen: [[[]]], render=True) -> [np.ndarray]:
        reduced = self._reduce(screen)

        # RGB
        reduced_red = reduced[:, :, 0]
        reduced_green = reduced[:, :, 1]
        reduced_blue = reduced[:, :, 2]

        mario_bm = self._detect_mario(reduced)
        monster_bm = self._detect_monsters(reduced)
        floor_bm = self._detect_floor(reduced)
        pipeshell_bm = self._detect_shell_or_pipe(reduced)

        '''
        # Test visuel
        display_rgb(
            np.uint8(
                np.where(
                    np.array(
                        [
                            [
                                np.full(3, True) if cond else np.full(3, False)
                                for cond in line
                            ]
                            for line in np.bitwise_or(mario_bm, monster_bm)
                        ]
                    ),
                    screen,
                    np.full((mario_bm.shape[0], mario_bm.shape[1], 3), np.zeros(3)),
                )
            )
        )
        '''

        return self._flatten_output(
            [
                reduced_red,
                reduced_green,
                reduced_blue,
                mario_bm,
                monster_bm,
                floor_bm,
                pipeshell_bm,
            ]
        )
