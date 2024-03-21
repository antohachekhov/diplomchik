import numpy as np

class Image:
    def __init__(self, img:np.ndarray=None, field:np.ndarray=None, mask:np.ndarray=None):
        if img is None:
            if field is not None or mask is not None:
                raise AttributeError("It is not possible to set mask or field in the absence of image")
        else:
            self._checkImage(img)
        self._img = img
        self._field = field
        if mask is not None:
            self._checkMask(img, mask)
        self._mask = mask

    @staticmethod
    def _checkImage(img:np.ndarray) -> bool:
        # Массив изображения должен иметь две оси (изображение в пространстве оттенков серого)
        if img.ndim != 2:
            raise ValueError('Image should be in grayscale')
        # Минимальная длина оси изображения не может быть меньше 12 (минимально допустимый размер измерительного окна)
        if min(img.shape) < 12:
            raise ValueError('Too small image')
        # Массив должен иметь значения в интервале [0, 255]
        if np.amin(img) < 0 or np.amax(img) > 255:
            raise ValueError('Invalid values in the array (values should be in range(0, 255))')
        return True

    @staticmethod
    def _checkMask(img, mask:np.ndarray) -> bool:
        if img is None:
            raise AttributeError("It is not possible to set mask in the absence of image")
        else:
            if mask.shape != img.shape:
                raise ValueError("Shape of input mask does not match shape of image")
            else:
                return True

    @property
    def mask(self) -> np.ndarray:
        return self._mask

    @mask.setter
    def mask(self, mask:np.ndarray):
        self._checkMask(self._img, mask)
        self._mask = mask

    @property
    def field(self) -> np.ndarray:
        return self._field

    @field.setter
    def field(self, field:np.ndarray):
        self._field = field

    @property
    def img(self) -> np.ndarray:
        return self._img