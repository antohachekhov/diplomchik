import numpy as np
from WindowProcessing import WindowProcessing
import FractalAnalysisInImage

class EstimationTransitionLayer:
    DefaultSettings = {
        'DBSCAN' : {
            'minSamplesInCluster': 8,
            'maxDistanceBetweenPointsInSingleCluster': -1
        },
        'WindowProcessing': {
            'fractalDimensionEstimationMethod': 'Prism',
            'parallelComputing': True,
            'windowSize': 32,
            'X-axisStep': 1,
            'Y-axisStep': 1,
            'numberWindowDivides': 2
        },
        'AnalysisOfFieldWithTwoComponents': {
            'minChangeFractalDimensionOfTwoTextures': 0.4,
            'widthOfMedianKernel': 21
        }
    }

    WindowFunctions = {
        'Prism': FractalAnalysisInImage.trianglePrism,
        'Cubes': FractalAnalysisInImage.cubes
    }

    def __init__(self, showStep:bool=False):
        self._img = None
        self._field = None
        self._maskImg = None
        self._showStep = showStep
        self._settings = EstimationTransitionLayer.DefaultSettings

    def clear(self):
        self._img = None
        self._field = None
        self._maskImg = None
        self._settings = EstimationTransitionLayer.DefaultSettings

    def setSettings(self, **kwargs):
        for key in kwargs:
            inter = set(kwargs[key]).intersection(self._settings[key])
            self._settings[key].update((keyIntersec, kwargs[key][keyIntersec]) for keyIntersec in inter)

    def setImage(self, image:np.ndarray):
        # Массив изображения должен иметь две оси (изображение в пространстве оттенков серого)
        if image.ndim != 2:
            raise ValueError('Image should be in grayscale')
        # Минимальная длина оси изображения не может быть меньше 12 (минимально допустимый размер измерительного окна)
        if min(image.shape) < 12:
            raise ValueError('Too small image')
        # Массив должен иметь значения в интервале [0, 255]
        if np.amin(image) < 0 or np.amax(image) > 255:
            raise ValueError('Invalid values in the array (values should be in range(0, 255))')
        self._img = image

    def setField(self, field:np.ndarray):
        # размер поля должен соответсвовать размеру изображения
        if field.shape != (int((self._img.shape[0] - self._settings['WindowProcessing']['windowSize']) /
                               self._settings['WindowProcessing']['Y-axisStep'] + 1),
                           int((self._img.shape[1] - self._settings['WindowProcessing']['windowSize']) /
                               self._settings['WindowProcessing']['X-axisStep'] + 1)):
            raise ValueError('The size of the field does not correspond to the size of the image and'
                             ' the selected window parameters')
        self._field = field

    def calculateField(self):
        winProcess = WindowProcessing(**self._settings['WindowProcessing'])
        subWindowsSizes = np.array([int((self._settings['WindowProcessing']['windowSize'] + (2 ** degree - 1)) / 2 ** degree)
                                            for degree in range(self._settings['WindowProcessing']['numberWindowDivides'] + 1)])
        self._field = winProcess.processing(self._img,
                                          EstimationTransitionLayer.WindowFunctions[
                                              self._settings['WindowProcessing']['fractalDimensionEstimationMethod']],
                                          subWindowsSizes)

    def getField(self):
        return self._field