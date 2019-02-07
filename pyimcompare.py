import cv2
import numpy as np
from typing import List, Union


class Image:
    def __init__(self, pt: list, height: int, width: int) -> None:
        self.topLeft: tuple = tuple(pt)
        self.topRight: tuple = (pt[0] + width, pt[1])
        self.bottomLeft: tuple = (pt[0], pt[1] + height)
        self.bottomRight: tuple = (pt[0] + width, pt[1] + height)
        self.center: tuple = ((self.topLeft[0] + self.topRight[0]) / 2, (self.topLeft[1] + self.bottomRight[1]) / 2)


def findImage(smallImg: Union[str, 'numpy.ndarray'], largeImg: Union[str, 'numpy.ndarray'] = 'screen',
              threshold: float = .7, maxResults: int = 1, delay: float = 0):

    if delay > 0:
        import time
        time.sleep(delay)

    if maxResults == 0:
        return []

    if type(smallImg) == str:
        smallImg = cv2.imread(smallImg)

    try:
        smallImg = cv2.cvtColor(smallImg, cv2.COLOR_BGR2GRAY)
    except cv2.error:
        raise FileNotFoundError(
            'The small image was not found or you have insufficient permissions to open it.') from None

    if largeImg == 'screen':
        import pyautogui
        largeImg = np.array(pyautogui.screenshot())
    elif type(largeImg) == str:
        largeImg = cv2.imread(largeImg)

    try:
        largeImg = cv2.cvtColor(largeImg, cv2.COLOR_BGR2GRAY)
    except cv2.error:
        raise FileNotFoundError(
            "The large image was not found or you have insufficient permisisons to open it.") from None

    height: int
    width: int

    height, width = smallImg.shape
    result: 'numpy.ndarray' = cv2.matchTemplate(largeImg, smallImg, cv2.TM_CCOEFF_NORMED)
    loc: tuple = np.where(result >= threshold)
    matches: List[Image] = []

    for pt in zip(*loc[::-1]):
        matches.append(Image(list(pt), height, width))

    if len(matches) == 0:
        return None
    elif len(matches) <= maxResults:
        return matches
    else:
        matches.clear()

    lowThreshold: float = threshold
    highThreshold: float = 1.0

    while True:
        increasedThreshold: float = 0.5 * (highThreshold - lowThreshold) + lowThreshold
        loc: tuple = np.where(result >= increasedThreshold)

        for pt in zip(*loc[::-1]):
            matches.append(Image(list(pt), height, width))

        if len(matches) == 0:
            highThreshold = increasedThreshold
        elif len(matches) <= maxResults:
            return matches
        else:
            lowThreshold = increasedThreshold
            matches.clear()
