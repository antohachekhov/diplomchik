from typing import Tuple, Any

from PIL import Image as PILImage


def getScale(imageName: str) -> float:
    exifData = PILImage.open(imageName).getexif()
    tags = exifData.items()._mapping[34118]

    width = 0.

    for row in tags.split("\r\n"):
        key_value = row.split(" = ")
        if len(key_value) == 2:
            if key_value[0] == "Width":
                width = key_value[1]
                break
    width_str = width.split(" ")

    width_unit = width_str[1]
    width_value = float(width_str[0]) # * measurementDict[width_unit[0]]
    ScaleCoef = width_value / float(exifData.items()._mapping[256])

    return ScaleCoef, width_unit[0]
