import numpy as np
import multiprocessing
import time
import os

class WindowProcessing:
    class WindowFunc(object):
        def __init__(self, func, *args):
            self._func = func
            self._args = args

        def __call__(self, window):
            return self._func(window, *self._args)

    def __init__(self, **kwargs):
        self._winSize = kwargs.get('windowSize', 32)
        self._dx = kwargs.get('X-axisStep', 1)
        self._dy = kwargs.get('Y-axisStep', 1)
        self._parallelComputing = kwargs.get('parallelComputing', False)

    def _generateWindows(self, img):
        """
        Генерация измерительных окон в каждой точке изображения

        :param img: ndarray
            Изображение
        """
        # win = np.zeros((self._win_size, self._win_size))
        # вычисляется размер массива измерительных окон
        shape = img.shape[:-2] + ((img.shape[-2] - self._winSize) // self._dy + 1,) + \
                ((img.shape[-1] - self._winSize) // self._dx + 1,) + (self._winSize, self._winSize)
        # вычисляется количество шагов измерительного окна
        strides = img.strides[:-2] + (img.strides[-2] * self._dy,) + \
                  (img.strides[-1] * self._dx,) + img.strides[-2:]
        return np.lib.stride_tricks.as_strided(img, shape=shape, strides=strides)

    def processing(self, img:np.ndarray, userWindowFunc, *args):
        """
        Оконная обработка изображения по переданной оконной функции
        :param img: ndarray
            Изображение
        :param userWindowFunc: function
            Оконная функция. В качестве первого аргумента функции должна принимать двухмерный массив - срез изображения
        :param args: list
            Список дополнительных аргументов функции
        :return: ndarray
            Массив результатов вычисления оконной функции для изображения
        """
        windowFunc = WindowProcessing.WindowFunc(userWindowFunc, *args)
        windows = self._generateWindows(img)
        print('Началось вычисление поля ФР')
        if self._parallelComputing:
            timer = time.time()
            with multiprocessing.Pool(processes=os.cpu_count()) as pool:
                result = [pool.map(windowFunc, rowWindows) for rowWindows in windows]
            print('Тест с параллельными вычислениями занял %.6f' % (time.time() - timer))
        else:
            timer = time.time()
            windowFuncVectorize = np.vectorize(windowFunc, signature='(w,w)->()')
            result = windowFuncVectorize(windows)
            print('Тест без параллельных вычислений занял %.6f' % (time.time() - timer))
        return np.asarray(result)