from ProgramEstimationTransitionLayer import EstimationTransitionLayer
from PIL import Image as PILImage
import numpy as np
import matplotlib.pyplot as plt
from ImagesMetadata import getScale
import threading

imageName = r"C:\Users\bortn\Desktop\diplomchik\analysis\new dataset\5\1-11\1-11.tif"
fieldPath = r"C:\Users\bortn\Desktop\diplomchik\analysis\new dataset\5\1-11\field_filt_prism_w30x1y1.csv"
#imageName = r"C:\Users\bortn\Desktop\diplomchik\analysis\new dataset\20\1-04\1-04.tif"
#fieldPath = r"C:\Users\bortn\Desktop\diplomchik\analysis\new dataset\20\1-04\field_field_prism_w30x1y1.csv"

def fieldToImage(field, show:bool=True):
    """
    Перевод поля фрактальных размерностей в пространство оттенков серого

    :param field: 2-D numpy массив
        Поле фрактальных размерностей.
    :param show: bool
        True, если необходимо вывести изображение визуализации поля

    :return imgOut: 2-D numpy массив
        Изображение-визуализация поля фрактальных размерностей
    """
    fieldConversionFunc = lambda x: int(255 * (x - 1) / 3) if x - 1.0 > 0.0 else 0
    fieldImg = np.vectorize(fieldConversionFunc)(field)
    if show:
        plt.imshow(fieldImg, cmap='gray')
        plt.show()
    return fieldImg

def inputImageWithCrop(path:str):
    InputImage = PILImage.open(imageName).convert('L')
    return np.asarray(InputImage.crop((0, 0, InputImage.size[0], InputImage.size[1] - 103)))

def inputField(path:str):
    return np.loadtxt(path, delimiter=";")


def main(image, field=None, mutex=None, nameSharedMemory=None, showStep=False):
    program = EstimationTransitionLayer(showStep, mutex=mutex, nameSharedMemory=nameSharedMemory)
    program.setSettings(WindowProcessing={'parallelComputing': True, 'windowSize': 30})
    scale = getScale(imageName)
    program.setImage(image, field)
    result = program.estimateTransitionLayer()
    print(result)
    # print(list(map(lambda d: d * scale, result)))
    program.clear()

if __name__ == "__main__":
    img = inputImageWithCrop(imageName)
    field = inputField(fieldPath)
    main(img, field, None, None, True)