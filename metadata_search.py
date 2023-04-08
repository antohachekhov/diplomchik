from PIL import Image as PILImage

def getScale(image_name):
    exif_data = PILImage.open(image_name).getexif()
    tags = exif_data.items()._mapping[34118]

    width = 0.
    measurement_dict = {
        "m": 1E-3,
        "µ": 1E-6,
        "n": 1E-9,
    }

    for row in tags.split("\r\n"):
        key_value = row.split(" = ")
        if len(key_value) == 2:
            if key_value[0] == "Width":
                width = key_value[1]
                break
    width_str = width.split(" ")

    width_unit = width_str[1]
    width_value = float(width_str[0]) * measurement_dict[width_unit[0]]
    ScaleCoef = width_value / float(exif_data.items()._mapping[256])

    return ScaleCoef