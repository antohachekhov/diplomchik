import numpy as np
import multiprocessing
import time
import os

class WindowProcessing:
    def __init__(self, **kwargs):
        self._winSize = kwargs.get('windowSize', 32)
        self._dx = kwargs.get('X-axisStep', 1)
        self._dy = kwargs.get('Y-axisStep', 1)
        self._parallelComputing = kwargs.get('parallelComputing', 2)

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

    def processing(self, img:np.ndarray, windowFunc, *args):
        """
        Оконная обработка изображения по переданной оконной функции
        :param img: ndarray
            Изображение
        :param windowFunc: function
            Оконная функция. В качестве первого аргумента функции должна принимать двухмерный массив - срез изображения
        :param args: list
            Список дополнительных аргументов функции
        :return: ndarray
            Массив результатов вычисления оконной функции для изображения
        """
        windows = self._generateWindows(img)
        result = list()
        if self._parallelComputing:
            print(os.cpu_count())
            timer = time.time()
            with multiprocessing.Pool(processes=os.cpu_count()) as pool:
                parallelResults = [[pool.apply_async(windowFunc, args=(window, *args))
                        for window in rowImage] for rowImage in windows]
                # parallelResults = []
                for rowRes in parallelResults:
                    result.append([resValue.get() for resValue in rowRes])
                # pool.close()
                # pool.join()
            print('Тест с параллельными вычислениями занял %.6f' % (time.time() - timer))
        else:
            timer = time.time()
            func_vec = np.vectorize(lambda window: windowFunc(window, *args), signature='(w,w)->()')
            result = func_vec(windows)
            print('Тест без параллельных вычислений занял %.6f' % (time.time() - timer))
        return np.asarray(result)