"""
Модуль оценки ширины переходного слоя на снимках сплава двух смесей металлов
"""

import time
import threading
import cv2 as cv
import numpy as np
import FractalAnalysisInImage
import matplotlib.pyplot as plt

from Image import Image
from scipy.signal import medfilt
from sklearn.cluster import DBSCAN
from sklearn.cluster import KMeans
from KneeLocatorModule import KneeLocator
from sklearn.mixture import GaussianMixture
from CountComponents import CountComponents
from WindowProcessing import WindowProcessing
from sklearn.neighbors import NearestNeighbors
from statsmodels.nonparametric.api import lowess
from MeasurementOnBinaryImage import MeasureObjects
from multiprocessing.shared_memory import SharedMemory


class EstimationTransitionLayer:
    """
    Класс алгоритма оценки ширины переходного слоя

    Атрибуты
    ----------
    _analyzedImg : ndarray, по умолчанию = None
        Анализируемое изображение

    _sharedMemory : SharedMemory, по умолчанию = None
        Указатель на общую память.

    _sharedMemoryBuffer : ndarray, по умолчанию = None
        Массив, хранящийся в общей памяти

    _show_step : bool, по умолчанию = False
        True, если необходимо отображать промежуточные результаты алгоритмов на экран.

    _settings : dict, по умолчанию = DefaultSettings
        Словарь настроек (параметров) алгоритмов.
    """
    DefaultSettings = {
        'DBSCAN': {
            'minSamplesInCluster': 8,
            'maxDistanceBetweenPointsInSingleCluster': -1
        },
        'WindowProcessing': {
            'fractalDimensionEstimationMethod': 'Prism',
            'parallelComputing': True,
            'windowSize': 30,
            'X-axisStep': 1,
            'Y-axisStep': 1,
            'numberWindowDivides': 2,
            'nComp': None
        },
        'AnalysisOfFieldWithTwoComponents': {
            'minChangeFractalDimensionOfTwoTextures': 0.4,
            'widthOfMedianKernel': 21,
            'columnarAnalysis': True,
            'LOWESS-frac': 0.1,
            'CoefWindow': 1
        },
        'AnalysisOfFieldWithThreeComponents': {
            'classification': 'k-means'
        },
        'Measure': {
            'CentralTendency': 'median'
        },
        'Multithreading': {
            'SignalForLoggingOfCurrentStep': None,
            'NameOfSharedMemory': None,
            'Mutex': None
        }
    }

    WindowFunctions = {
        'Prism': FractalAnalysisInImage.trianglePrism,
        'Cubes': FractalAnalysisInImage.cubes
    }

    CentralTendency = {
        'median': np.median,
        'mean': np.mean
    }

    Colors = {
        -1: (139, 0, 139),  # темно-пурпурный
        0: (0, 255, 255),  # желтый
        1: (180, 119, 31),  # голубой
        2: tuple(reversed((44, 160, 44))),  # зелёный
        3: tuple(reversed((214, 39, 40))),  # красный
        4: tuple(reversed((148, 103, 189))),  # пурпурный
        5: tuple(reversed((140, 86, 75))),  # коричневый
        6: tuple(reversed((227, 119, 194))),  # розовый
        7: tuple(reversed((188, 189, 34))),  # оливковый
        8: tuple(reversed((23, 190, 207))),  # циан
        9: (0, 0, 128),  # бордовый
        10: (80, 127, 255),  # коралл
        11: (0, 165, 255),  # апельсин
        12: (212, 255, 127),  # Аквамарин
        13: (0, 252, 124),  # газон зеленый
        14: (114, 128, 250),  # лосось
        15: (128, 0, 0),  # флот
        16: (50, 205, 154),  # желтый зеленый
        17: (180, 105, 255),  # ярко-розовый
        18: (130, 0, 75),  # индиго
        19: (255, 0, 0),  # синий
        20: (14, 127, 255),  # оранжевый
        21: (50, 205, 50),  # зеленый лайм
        22: (140, 230, 240),  # хаки
        23: (34, 139, 34),  # зеленый лес
        24: (154, 250, 0),  # средний весенний
        25: (0, 215, 255),  # золото
        26: (237, 149, 100),  # кукуруза цветок синий
        27: (144, 238, 144),  # светло-зеленый
        28: (255, 255, 0),  # аква
        29: (113, 179, 60),  # средний морской зеленый
    }

    def __init__(self, showStep: bool = False):
        """
        Конструктор класса
        :param showStep : bool, по умолчанию = False.
            Флаг об отображении промежуточных результатов на экране.
        """
        self._analyzedImg = None
        self._showStep = showStep
        self._settings = EstimationTransitionLayer.DefaultSettings
        self._sharedMemory = None
        self._sharedMemoryBuffer = None

    def clear(self, clearSettings=True):
        """
        Очистка данных
        :param clearSettings: bool, по умолчанию = True
            Флаг о необходимости приведения настроек к виду по умолчанию
        :return: None
        """
        self._analyzedImg = None
        if clearSettings:
            self._settings = EstimationTransitionLayer.DefaultSettings
        self._sharedMemory = None
        if self._settings['Multithreading']['Mutex'] is not None:
            self._settings['Multithreading']['Mutex'].release()
            print(f'{threading.current_thread().name} освободил ресурс')

    def setSettings(self, **kwargs):
        """
        Установка настроек
        :param kwargs: dict
            Входные настройки
        :return: None
        """
        for key in kwargs:
            inter = set(kwargs[key]).intersection(self._settings[key])
            self._settings[key].update((keyIntersec, kwargs[key][keyIntersec]) for keyIntersec in inter)

    def _checkField(self, image:np.ndarray, field: np.ndarray) -> bool:
        """
        Проверка входного ПФР на соответствие размеров изображения и измерительного окна
        :param image: ndarray
            Изображение
        :param field: ndarray
            ПФР
        :return: bool
            True - если проверка прошла успешно, иначе - False
        """
        # размер поля должен соответсвовать размеру изображения
        if field.shape != (int((image.shape[0] - self._settings['WindowProcessing']['windowSize']) /
                               self._settings['WindowProcessing']['Y-axisStep'] + 1),
                           int((image.shape[1] - self._settings['WindowProcessing']['windowSize']) /
                               self._settings['WindowProcessing']['X-axisStep'] + 1)):
            raise ValueError('The size of the field does not correspond to the size of the image and'
                             ' the selected window parameters')
        return True

    def setImage(self, image: np.ndarray, field: np.ndarray = None, mask: np.ndarray = None):
        """
        Ввод изображения
        :param image: ndarray
            Входное изображение
        :param field: ndarray, по умолчанию = None
            ПФР
        :param mask: ndarray, по умолчанию = None
            Маска изображения
        :return: None
        """
        if field is not None:
            self._checkField(image, field)
        self._analyzedImg = Image(image, field, mask)

    def getField(self):
        """
        Вывод ПФР
        :return: ndarray
            ПФР
        """
        return self._analyzedImg.field

    def setField(self, field: np.ndarray):
        """
        Ввод ПФР
        :param field: ndarray
            Входное ПФР
        :return: None
        """
        if self._analyzedImg is None:
            raise AttributeError("It is not possible to set field in the absence of image")
        self._checkField(field)
        self._analyzedImg.field = field

    def getMask(self):
        """
        Получить маску
        :return: ndarray
            Маска для введенного изображения
        """
        return self._analyzedImg.mask

    def setMask(self, mask: np.ndarray):
        """
        Ввод маски
        :param mask: ndarray
            Маска для введённого изображения
        :return: None
        """
        self._analyzedImg.mask = mask

    def _logging(self, message:str):
        """
        Вывод сообщений об исполнении алгоритмов
        :param message: str
            Сообщение
        :return: None
        """
        if self._settings['Multithreading']['SignalForLoggingOfCurrentStep'] is not None:
            # вывод сообщения через сигнал в основной поток
            self._settings['Multithreading']['SignalForLoggingOfCurrentStep'].emit(message)
        else:
            # вывод сообщения на экран
            print(message)

    def _calculateField(self):
        """
        Вычисление ПФР
        :return: None
        """
        # Инициализация оконной обработки
        winProcess = WindowProcessing(**self._settings['WindowProcessing'])
        # Определение размеров субокон
        subWindowsSizes = np.array(
            [int((self._settings['WindowProcessing']['windowSize'] + (2 ** degree - 1)) / 2 ** degree)
             for degree in range(self._settings['WindowProcessing']['numberWindowDivides'] + 1)])

        self._logging(
            f"Шаг 1/8: Вычисление ПФР ({'параллельно' if self._settings['WindowProcessing']['parallelComputing'] else 'последовательно'})")
        # Вычисление ПФР с помощью оконной обработки с заданной оконной функцией
        self._analyzedImg.field = winProcess.processing(self._analyzedImg.img,
                                                        EstimationTransitionLayer.WindowFunctions[
                                                            self._settings['WindowProcessing'][
                                                                'fractalDimensionEstimationMethod']],
                                                        subWindowsSizes)

    @staticmethod
    def _findExtr(data: np.ndarray):
        """
        Поиск локальных экстремумов и величин изменения между ними

        :param data: 1-D numpy массив.
            Массив, в котором необходимо найти экстремумы

        :return ampl : 1-D numpy массив.
            Массив значений изменения значения переменной между локальными экстремумами

        :return extr : 1-D numpy массив.
            Массив координат локальных экстремумов
        """
        extr = np.empty(0, dtype=np.uint32)
        ampl = np.empty(0, dtype=np.float64)
        i = 0

        # флаг-переменная. Истина, если впереди локальный максимум
        toMax = False

        # проход по значениям, пока значения не станут изменяться
        while data[i + 1] == data[i] and i + 1 != data.size:
            i += 1

        # начало изменения заносится в массив экстремумов
        extr = np.append(extr, [i])

        if i + 1 != data.size:
            # если следующее значение больше
            if data[i + 1] > data[i]:
                # впереди локальный максимум
                toMax = True
            else:
                # иначе локальный минимум
                toMax = False

        # пока не достигнут конец массива
        while i + 1 != data.size:
            # если впереди локальный максимум
            if toMax:
                # проход по значениям, пока значения не начнут уменьшатся
                while i + 1 != data.size and data[i + 1] >= data[i]:
                    i += 1
                # если значения уменьшаются, впереди локальный минимум
                toMax = False
            else:
                # проход вперёд, пока значения не начнут увеличиваться
                while i + 1 != data.size and data[i + 1] <= data[i]:
                    i += 1
                # если значения увеличиваются, впереди локальный максимум
                toMax = True
            # разность между локальными экстремумами заносится в массив величин изменений
            ampl = np.append(ampl, [abs(data[i] - data[extr[-1]])])
            # координата найденного локального экстремума заносится в массив координат
            extr = np.append(extr, [i])
        return ampl, extr

    def _findExtrWithDiff(self, data):
        """
        Поиск локальных экстремумов и величин изменения между ними

        :param data: 1-D numpy массив.
            Массив, в котором необходимо найти экстремумы

        :return ampl : 1-D numpy массив.
            Массив значений изменения значения переменной между локальными экстремумами

        :return extr : 1-D numpy массив.
            Массив координат локальных экстремумов

        :return A : list
            Множество точек A

        :return B : list
            Множество точек B
        """
        extr = np.empty(0, dtype=np.uint32)
        ampl = np.empty(0, dtype=np.float64)

        A = np.empty(0, dtype=np.uint32)
        B = np.empty(0, dtype=np.uint32)

        i = 0

        # флаг-переменная. Истина, если впереди локальный максимум
        toMax = False

        # проход по значениям, пока значения не станут изменяться
        while data[i + 1] == data[i] and i + 1 != data.size:
            i += 1

        # начало изменения заносится в массив экстремумов
        extr = np.append(extr, [i])

        if i + 1 != data.size:
            # если следующее значение больше
            if data[i + 1] > data[i]:
                # впереди локальный максимум
                toMax = True
            else:
                # иначе локальный минимум
                toMax = False

        findMax = False

        # пока не достигнут конец массива
        while i + 1 != data.size:
            # если впереди локальный максимум
            if toMax:
                # проход по значениям, пока значения не начнут уменьшатся
                while i + 1 != data.size and data[i + 1] >= data[i]:
                    i += 1
                # если значения уменьшаются, впереди локальный минимум
                toMax = False
                if findMax:
                    B = np.append(B, [i])
                    findMax = False
            else:
                # проход вперёд, пока значения не начнут увеличиваться
                while i + 1 != data.size and data[i + 1] <= data[i]:
                    i += 1
                # если значения увеличиваются, впереди локальный максимум
                toMax = True

            # Разность между соседними экстремумами
            curDiff = abs(data[i] - data[extr[-1]])
            # Проверки превышения разности на заданный порог
            if curDiff >= self._settings['AnalysisOfFieldWithTwoComponents']['minChangeFractalDimensionOfTwoTextures']:
                # если впереди локальный максимум
                if not toMax:
                    if len(extr) >= 2:
                        B = np.append(B, [extr[-2]])
                        A = np.append(A, [i])
                else:
                    A = np.append(A, [extr[-1]])
                    findMax = True

            # разность между локальными экстремумами заносится в массив величин изменений
            ampl = np.append(ampl, [curDiff])
            # координата найденного локального экстремума заносится в массив координат
            extr = np.append(extr, [i])

        if findMax:
            # если для последнего локального минимума не был найден локальный максимум
            A = A[:-1]

        return ampl, extr, A, B

    def _putInSharedMemory(self, data:np.ndarray, description: str = None):
        """
        Вывод данных в общую память
        :param data: ndarray
            Данные
        :param description: str
            Описание данных
        :return: None
        """
        self._sharedMemoryBuffer[:] = data
        print(f'{threading.current_thread().name} положил в ОП{' ' + description if description else ''}')
        self._settings['Multithreading']['Mutex'].release()
        print(f'{threading.current_thread().name} освободил мутекс')
        time.sleep(1)
        print(f'{threading.current_thread().name} ожидает доступ к мутексу')
        self._settings['Multithreading']['Mutex'].acquire()
        print(f'{threading.current_thread().name} получил доступ к мутексу')

    def _analysisFieldWithTwoComponents(self):
        """
        Сегментация переходного слоя по сильным изменения поля

        """
        # массивы точек начала и конца сильных изменений соответственно
        changeBegin = np.array([0, 0], dtype=np.uint32)
        changeEnd = np.array([0, 0], dtype=np.uint32)

        self._logging('Шаг 3/8 (алгоритм 1): Определение точек А и В')
        # проход по каждому столбцу в поле
        for n, sliceOfField in enumerate(
                self._analyzedImg.field.T if self._settings['AnalysisOfFieldWithTwoComponents'][
                    'columnarAnalysis'] else self._analyzedImg.field):
            # медианная фильтрация
            filterSlice = medfilt(sliceOfField,
                                  kernel_size=self._settings['AnalysisOfFieldWithTwoComponents']['widthOfMedianKernel'])

            # # координаты локальных экстремумов и разность значений между ними
            # amplMF, extrMF = self._findExtr(filterSlice)
            #
            # # Выделение областей, в которых разность между экстремумами больше,
            # # чем величина изменения фрактальной размерности на границе двух текстур
            # indexPeak = np.where(
            #     amplMF > self._settings['AnalysisOfFieldWithTwoComponents']['minChangeFractalDimensionOfTwoTextures'])
            # if indexPeak[0].size > 0:
            #     for j in indexPeak[0]:
            #         # области на границе изображения не берутся во внимание,
            #         # так как на границах поле искажено
            #         if j != 0 and j < extrMF.shape[0] - 1:
            #             if 0 < extrMF[j] < self._analyzedImg.field.shape[
            #                 self._settings['AnalysisOfFieldWithTwoComponents']['columnarAnalysis']] - 1 and extrMF[
            #                 j + 1] < self._analyzedImg.field.shape[
            #                 self._settings['AnalysisOfFieldWithTwoComponents']['columnarAnalysis']] - 1:
            #                 changeBegin = np.vstack([changeBegin, np.array([n, extrMF[j]])])
            #                 changeEnd = np.vstack([changeEnd, np.array([n, extrMF[j + 1]])])

            # Определение множеств A и B
            _, _, A, B = self._findExtrWithDiff(filterSlice)
            for a, b in zip(A, B):
                changeBegin = np.vstack([changeBegin, np.array([n, a])])
                changeEnd = np.vstack([changeEnd, np.array([n, b])])

        changeBegin = np.delete(changeBegin, 0, axis=0)
        changeEnd = np.delete(changeEnd, 0, axis=0)

        # перевод точек поля в размерность изображения
        radius = int(self._settings['WindowProcessing'][
                    'windowSize'] + (2 ** self._settings['WindowProcessing']['numberWindowDivides'] - 1)) / 2 ** self._settings['WindowProcessing']['numberWindowDivides'] / self._settings['AnalysisOfFieldWithTwoComponents']['CoefWindow']
        for point1, point2 in zip(changeBegin, changeEnd):
            point1[0] = int(point1[0] * self._settings['WindowProcessing']['X-axisStep'] + self._settings['WindowProcessing'][
                    'windowSize'] / 2)
            point2[0] = int(point2[0] * self._settings['WindowProcessing']['X-axisStep'] + self._settings['WindowProcessing'][
                    'windowSize'] / 2)

            if point1[1] < point2[1]:
                point1[1] = int(point1[1] * self._settings['WindowProcessing']['X-axisStep'] + self._settings['WindowProcessing'][
                    'windowSize'] / 2) + radius
                point2[1] = int(point2[1] * self._settings['WindowProcessing']['X-axisStep'] + self._settings['WindowProcessing'][
                    'windowSize'] / 2) - radius
            else:
                point1[1] = int(point1[1] * self._settings['WindowProcessing']['X-axisStep'] + self._settings['WindowProcessing'][
                    'windowSize'] / 2) - radius
                point2[1] = int(point2[1] * self._settings['WindowProcessing']['X-axisStep'] + self._settings['WindowProcessing'][
                    'windowSize'] / 2) + radius

        # for point in changeBegin:
        #     point[0] = int(
        #         point[0] * self._settings['WindowProcessing']['X-axisStep'] + self._settings['WindowProcessing'][
        #             'windowSize'] / 2)
        #     point[1] = int(
        #         point[1] * self._settings['WindowProcessing']['Y-axisStep'] + self._settings['WindowProcessing'][
        #             'windowSize'] / 2)
        #
        # for point in changeEnd:
        #     point[0] = int(
        #         point[0] * self._settings['WindowProcessing']['X-axisStep'] + self._settings['WindowProcessing'][
        #             'windowSize'] / 2)
        #     point[1] = int(
        #         point[1] * self._settings['WindowProcessing']['Y-axisStep'] + self._settings['WindowProcessing'][
        #             'windowSize'] / 2)

        if self._showStep:
            plt.imshow(cv.cvtColor(self._analyzedImg.img, cv.COLOR_GRAY2RGB))
            plt.scatter(changeBegin[:, 0], changeBegin[:, 1], c='red', s=1)
            plt.scatter(changeEnd[:, 0], changeEnd[:, 1], c='yellow', s=1)
            plt.show()

        # Кладём в общую память изображение с распределением точек
        if self._settings['Multithreading']['Mutex'] is not None:
            tempImg = cv.cvtColor(self._analyzedImg.img, cv.COLOR_GRAY2BGR)
            for b, e in zip(changeBegin, changeEnd):
                tempImg = cv.circle(tempImg, (b[0], b[1]), radius=3, color=(0, 0, 255), thickness=-1)
                tempImg = cv.circle(tempImg, (e[0], e[1]), radius=3, color=(0, 255, 255), thickness=-1)
            self._putInSharedMemory(cv.cvtColor(tempImg, cv.COLOR_BGR2RGB), 'распределение точек А и В')

        self._logging('Шаг 4/8 (алгоритм 1): DBSCAN-кластеризация')
        # Оценка параметра eps алгоритма DBSCAN
        nearestNeighbors = NearestNeighbors(n_neighbors=11)
        neighbors = nearestNeighbors.fit(changeBegin)
        distances, indices = neighbors.kneighbors(changeBegin)
        distances = np.sort(distances[:, 10], axis=0)
        i = np.arange(len(distances))
        knee = KneeLocator(i, distances, S=1, curve='convex', direction='increasing', interp_method='polynomial')

        # если пользователем не задано другое значение distance
        if self._settings['DBSCAN']['maxDistanceBetweenPointsInSingleCluster'] == -1.:
            self._settings['DBSCAN']['maxDistanceBetweenPointsInSingleCluster'] = 1. * distances[knee.knee]

        # кластеризация точек, соответствующих началу изменения
        dbscanCluster = DBSCAN(eps=self._settings['DBSCAN']['maxDistanceBetweenPointsInSingleCluster'],
                               min_samples=self._settings['DBSCAN']['minSamplesInCluster'])
        dbscanCluster.fit(changeBegin)

        # Кладём в общую память изображение с результатами кластеризации
        if self._settings['Multithreading']['Mutex'] is not None:
            tempImg = cv.cvtColor(self._analyzedImg.img, cv.COLOR_GRAY2BGR)
            for i, z in enumerate(zip(changeBegin, changeEnd)):
                b, e = z
                tempImg = cv.circle(tempImg, (b[0], b[1]), radius=3,
                                    color=EstimationTransitionLayer.Colors[dbscanCluster.labels_[i]], thickness=-1)
                tempImg = cv.circle(tempImg, (e[0], e[1]), radius=3,
                                    color=EstimationTransitionLayer.Colors[dbscanCluster.labels_[i]], thickness=-1)
            self._putInSharedMemory(cv.cvtColor(tempImg, cv.COLOR_BGR2RGB), 'распределение точек после кластеризации')

        if self._showStep:
            plt.imshow(cv.cvtColor(self._analyzedImg.img, cv.COLOR_GRAY2RGB))
            plt.scatter(changeBegin[:, 0], changeBegin[:, 1], c=dbscanCluster.labels_, s=15)
            plt.scatter(changeEnd[:, 0], changeEnd[:, 1], c=dbscanCluster.labels_, s=15)
            plt.show()

        self._logging('Шаг 5/8 (алгоритм 1): Фильтрация шумов')
        # получение списка номеров кластеров и их размеров
        uniqLabels, labelCount = np.unique(dbscanCluster.labels_, return_counts=True)

        # средний размер кластеров, не считая кластер шума
        labelCountMean = np.mean(labelCount[1:])

        # массивы для точек из не отброшенных кластеров
        clusteredChangeBegin = np.array([0, 0], dtype=np.uint32)
        clusteredChangeEnd = np.array([0, 0], dtype=np.uint32)
        filteredLabels = np.empty(0, dtype=np.uint8)

        self._logging('Шаг 6/8 (алгоритм 1): Фильтрация мелких кластеров')
        # сохранение точек, размер кластера которых больше среднего размера всех кластеров
        for i in range(1, uniqLabels.shape[0]):
            if labelCount[i] >= labelCountMean:
                clusteredChangeBegin = np.vstack([clusteredChangeBegin,
                                                  changeBegin[dbscanCluster.labels_ == uniqLabels[i]]])
                clusteredChangeEnd = np.vstack([clusteredChangeEnd,
                                                changeEnd[dbscanCluster.labels_ == uniqLabels[i]]])
                filteredLabels = np.append(filteredLabels, [i] * labelCount[i])

        clusteredChangeBegin = np.delete(clusteredChangeBegin, 0, axis=0)
        clusteredChangeEnd = np.delete(clusteredChangeEnd, 0, axis=0)

        # Кладём в общую память изображение с результатами кластеризации после фильтрации
        if self._settings['Multithreading']['Mutex'] is not None:
            tempImg = cv.cvtColor(self._analyzedImg.img, cv.COLOR_GRAY2BGR)
            for i, z in enumerate(zip(clusteredChangeBegin, clusteredChangeEnd)):
                b, e = z
                tempImg = cv.circle(tempImg, (b[0], b[1]), radius=3,
                                    color=EstimationTransitionLayer.Colors[filteredLabels[i]], thickness=-1)
                tempImg = cv.circle(tempImg, (e[0], e[1]), radius=3,
                                    color=EstimationTransitionLayer.Colors[filteredLabels[i]], thickness=-1)
            self._putInSharedMemory(cv.cvtColor(tempImg, cv.COLOR_BGR2RGB), 'распределение точек после кластеризации и фильтрации')

        if self._showStep:
            plt.imshow(cv.cvtColor(self._analyzedImg.img, cv.COLOR_GRAY2RGB))
            plt.scatter(clusteredChangeBegin[:, 0], clusteredChangeBegin[:, 1], c=filteredLabels, s=10)
            plt.scatter(clusteredChangeEnd[:, 0], clusteredChangeEnd[:, 1], c=filteredLabels, s=10)
            plt.show()

        if self._showStep:
            # перевод изображения в цветовое пространство RGB
            imgRGB = cv.cvtColor(self._analyzedImg.img, cv.COLOR_GRAY2RGB)
            fig, axis = plt.subplots(2, 2)

            axis[0, 0].imshow(imgRGB)
            axis[0, 0].scatter(changeBegin[:, 0], changeBegin[:, 1], c='red', s=1)
            axis[0, 0].scatter(changeEnd[:, 0], changeEnd[:, 1], c='red', s=1)
            axis[0, 0].set_title("Image and amplitude's points")

            axis[0, 1].set_title(
                'Knee Point = ' + str(self._settings['DBSCAN']['maxDistanceBetweenPointsInSingleCluster']))
            axis[0, 1].plot(knee.x, knee.y, "b", label="data")
            axis[0, 1].vlines(knee.knee, plt.ylim()[0], plt.ylim()[1], linestyles="--", label="knee/elbow")
            axis[0, 1].legend(loc="best")

            axis[1, 0].scatter(changeBegin[:, 0], changeBegin[:, 1], c=dbscanCluster.labels_, s=15)
            axis[1, 0].scatter(changeEnd[:, 0], changeEnd[:, 1], c=dbscanCluster.labels_, s=15)
            axis[1, 0].set_title('DBSCAN eps=' + str(
                self._settings['DBSCAN']['maxDistanceBetweenPointsInSingleCluster']) + ' min_samples=' + str(
                self._settings['DBSCAN']['minSamplesInCluster']))

            axis[1, 1].scatter(clusteredChangeBegin[:, 0], clusteredChangeBegin[:, 1], c=filteredLabels, s=10)
            axis[1, 1].scatter(clusteredChangeEnd[:, 0], clusteredChangeEnd[:, 1], c=filteredLabels, s=10)
            axis[1, 1].set_title('[Filtered] DBSCAN eps=' + str(
                self._settings['DBSCAN']['maxDistanceBetweenPointsInSingleCluster']) + ' m_samp=' + str(
                self._settings['DBSCAN']['minSamplesInCluster']))

            plt.show()
            plt.close()

        self.listDistancesInEveryPoint = []

        self._logging('Шаг 7/8 (алгоритм 1): LOWESS-регрессия')
        tempImg = cv.cvtColor(self._analyzedImg.img, cv.COLOR_GRAY2BGR)
        # проход по каждому кластеру
        for i in np.unique(filteredLabels):
            x = clusteredChangeBegin[filteredLabels == i, 0]
            y1 = clusteredChangeBegin[filteredLabels == i, 1]
            y2 = clusteredChangeEnd[filteredLabels == i, 1]

            # строится регрессионная модель для точек, соответствующих началу переходного изменения поля
            z1 = lowess(y1, x, frac=self._settings['AnalysisOfFieldWithTwoComponents']['LOWESS-frac'])
            # строится регрессионная модель для точек, соответствующих концу переходного изменения поля
            z2 = lowess(y2, x, frac=self._settings['AnalysisOfFieldWithTwoComponents']['LOWESS-frac'])

            if self._settings['Multithreading']['Mutex']is not None:
                cv.polylines(tempImg, [z1.astype(np.int32).reshape((-1, 1, 2))], False, (255, 0, 0), 2)
                cv.polylines(tempImg, [z2.astype(np.int32).reshape((-1, 1, 2))], False, (255, 0, 0), 2)


            # округляем точки до целых, чтобы поставить им в соответствие пиксели изображения
            pointsBeg = np.around(z1).astype(np.int32)
            pointsBeg = pointsBeg.reshape((-1, 1, 2))
            pointsEnd = np.around(z2).astype(np.int32)
            pointsEnd = np.flipud(pointsEnd).reshape((-1, 1, 2))
            pts = np.concatenate([pointsBeg, pointsEnd, pointsBeg[0].reshape((-1, 1, 2))])

            # рисуем на маске линии начала и конца изменения и закрашиваем область между ними
            cv.fillPoly(self._analyzedImg.mask, [pts], 255)

        # Кладём в общую память изображение с кривыми
        if self._settings['Multithreading']['Mutex'] is not None:
            self._putInSharedMemory(cv.cvtColor(tempImg, cv.COLOR_BGR2RGB), 'кривые')

    def _segmentForDistribution(self, img, intensity):
        """
        Сегментация областей изображения, соответствующих заданному классу области

        :param img: 2-D numpy массив.
            Изображение состоящее из n цветов в пространстве оттенков серого.
            Каждый цвет - отдельный класс области.

        :param intensity: int
            Значение интенсивности цвета, который соответствует необходимым для сегментации областям.

        :return img2: 2-D numpy массив.
            Бинарная маска, объекты на которой соответствуют областям принадлежащим заданному классу
        """
        # выделение и бинаризация областей с заданной интенсивностью цвета
        self._logging('Шаг 4/8 (алгоритм 2): Пороговая бинаризация')
        maskByIntensity = cv.inRange(img, intensity, intensity + 1)

        # Кладём в общую память изображение с пороговой бинаризацией
        if self._settings['Multithreading']['Mutex'] is not None:
            tempMask = np.zeros(self._analyzedImg.img.shape, dtype=np.uint8)
            tempMask[int(self._settings['WindowProcessing']['windowSize'] / 2): -int(
                self._settings['WindowProcessing']['windowSize'] / 2) + 1,
            int(self._settings['WindowProcessing']['windowSize'] / 2): -int(
                self._settings['WindowProcessing']['windowSize'] / 2) + 1] = maskByIntensity
            self._putInSharedMemory(self._maskOnImage(tempMask), 'пороговую бинаризацию')

        self._logging('Шаг 5/8 (алгоритм 2): Морфологические операции Открытие и Закрытие')
        # "открытие" изображения (удаление мелких объектов) с эллиптическим ядром 7х7
        kernel = cv.getStructuringElement(cv.MORPH_RECT, (7, 7))
        maskAfterOpen = cv.morphologyEx(maskByIntensity, cv.MORPH_OPEN, kernel)

        if self._settings['Multithreading']['Mutex'] is not None:
            tempMask = np.zeros(self._analyzedImg.img.shape, dtype=np.uint8)
            tempMask[int(self._settings['WindowProcessing']['windowSize'] / 2): -int(
                self._settings['WindowProcessing']['windowSize'] / 2) + 1,
            int(self._settings['WindowProcessing']['windowSize'] / 2): -int(
                self._settings['WindowProcessing']['windowSize'] / 2) + 1] = maskAfterOpen
            self._putInSharedMemory(self._maskOnImage(tempMask), 'результат операции Открытие')

        # "закрытие" изображения (заполнение мелких пропусков в объектах) с эллиптическим ядром 5х5
        kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE, (5, 5))
        maskAfterClose = cv.morphologyEx(maskAfterOpen, cv.MORPH_CLOSE, kernel)

        if self._settings['Multithreading']['Mutex'] is not None:
            tempMask = np.zeros(self._analyzedImg.img.shape, dtype=np.uint8)
            tempMask[int(self._settings['WindowProcessing']['windowSize'] / 2): -int(
                self._settings['WindowProcessing']['windowSize'] / 2) + 1,
            int(self._settings['WindowProcessing']['windowSize'] / 2): -int(
                self._settings['WindowProcessing']['windowSize'] / 2) + 1] = maskAfterClose
            self._putInSharedMemory(self._maskOnImage(tempMask), 'результат операции Закрытие')

        self._logging('Шаг 6/8 (алгоритм 2): Фильтрация мелких объектов')
        # выделение объектов на изображении и составление статистики
        contours, hierarchy = cv.findContours(maskAfterClose.copy(), cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
        cv.drawContours(maskAfterClose, contours, -1, 100, 2, cv.LINE_AA, hierarchy, 3)
        nb_components, output, stats, centroids = cv.connectedComponentsWithStats(maskAfterClose, 4, cv.CV_32S)

        # вычисление средней площади всех объектов
        meanS = np.mean(stats[1:, -1])

        # массив для маски с областями для сегментации
        maskAfterFilter = np.zeros(output.shape, dtype=np.uint8)

        # заполнение маски
        for i in range(1, nb_components):
            # заносятся только те объекты, площадь которых больше средней площади всех объектов
            if stats[i, -1] > meanS:
                maskAfterFilter[output == i] = 255

        if self._settings['Multithreading']['Mutex'] is not None:
            tempMask = np.zeros(self._analyzedImg.img.shape, dtype=np.uint8)
            tempMask[int(self._settings['WindowProcessing']['windowSize'] / 2): -int(
                self._settings['WindowProcessing']['windowSize'] / 2) + 1,
            int(self._settings['WindowProcessing']['windowSize'] / 2): -int(
                self._settings['WindowProcessing']['windowSize'] / 2) + 1] = maskAfterFilter
            self._putInSharedMemory(self._maskOnImage(tempMask), 'результат после фильтрации')

        self._logging('Шаг 7/8 (алгоритм 2): Морфологическая операция Дилатация')
        # наращивание объектов
        # radius = int(self._settings['WindowProcessing']['windowSize'] / 2 if self._settings['WindowProcessing'][
        #                                                                          'windowSize'] / 2 % 2 == 1 else
        #              self._settings['WindowProcessing']['windowSize'] / 2 - 1)
        # kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE, (radius, radius))
        # maskAfterDilate = cv.dilate(maskAfterFilter, kernel)
        radius = int(self._settings['WindowProcessing']['windowSize'] / 3 if self._settings['WindowProcessing'][
                                                                                 'windowSize'] / 3 % 2 == 1 else
                     self._settings['WindowProcessing']['windowSize'] / 3 - 1)
        radius = int(self._settings['WindowProcessing']['windowSize'] if self._settings['WindowProcessing'][
                                                                                  'windowSize'] % 2 == 1 else
                      self._settings['WindowProcessing']['windowSize'] - 1)
        maskAfterBlur = cv.GaussianBlur(maskAfterFilter, (radius, radius), 0)

        if self._settings['Multithreading']['Mutex'] is not None:
            tempMask = np.zeros(self._analyzedImg.img.shape, dtype=np.uint8)
            tempMask[int(self._settings['WindowProcessing']['windowSize'] / 2): -int(
                self._settings['WindowProcessing']['windowSize'] / 2) + 1,
            int(self._settings['WindowProcessing']['windowSize'] / 2): -int(
                self._settings['WindowProcessing']['windowSize'] / 2) + 1] = maskAfterBlur
            self._putInSharedMemory(self._maskOnImage(tempMask), 'результат операции Дилатация')

        if self._showStep:
            plt.subplot(2, 3, 1)
            plt.imshow(img, cmap='gray')
            plt.title('Original')
            plt.xticks([])
            plt.yticks([])

            plt.subplot(2, 3, 2)
            plt.imshow(maskByIntensity, cmap='gray')
            plt.title('Range')
            plt.xticks([])
            plt.yticks([])

            plt.subplot(2, 3, 3)
            plt.imshow(maskAfterOpen, cmap='gray')
            plt.title('Open')
            plt.xticks([])
            plt.yticks([])

            plt.subplot(2, 3, 4)
            plt.imshow(maskAfterClose, cmap='gray')
            plt.title('Close')
            plt.xticks([])
            plt.yticks([])

            plt.subplot(2, 3, 5)
            plt.imshow(maskAfterFilter, cmap='gray')
            plt.title('Sort')
            plt.xticks([])
            plt.yticks([])

            plt.subplot(2, 3, 6)
            plt.imshow(maskAfterBlur, cmap='gray')
            plt.title('Blur&Thresh')
            plt.xticks([])
            plt.yticks([])

            plt.show()

        return maskAfterBlur

    def _analysisFieldWithThreeComponents(self):
        """
        Сегментация переходного слоя через EM-классификацию
        """
        self._logging('Шаг 3/8 (алгоритм 2): Кластеризация по алгоритму k-средних')
        # классификация значений поля по EM-алгоритму с тремя компонентами в смеси
        X = np.expand_dims(self._analyzedImg.field.flatten(), 1)
        kmeans = KMeans(n_clusters=3, random_state=0, n_init=5).fit(X)
        predict = kmeans.labels_
        means = kmeans.cluster_centers_
        # model = GaussianMixture(n_components=3, covariance_type='full')
        # model.fit(X)
        # predict = model.predict(X)
        # means = model.means_
        fieldPredict = np.resize(predict, self._analyzedImg.field.shape)

        # перевод поля значений классов в пространство оттенков серого
        fieldPredImg = fieldPredict / 2 * 255
        fieldPredImg = fieldPredImg.astype(np.uint8)

        # Кладём в общую память изображение с результатами классификации
        if self._settings['Multithreading']['Mutex'] is not None:
            tempMask = np.zeros(self._analyzedImg.img.shape, dtype=np.uint8)
            tempMask[int(self._settings['WindowProcessing']['windowSize'] / 2): -int(
                self._settings['WindowProcessing']['windowSize'] / 2) + 1,
            int(self._settings['WindowProcessing']['windowSize'] / 2): -int(
                self._settings['WindowProcessing']['windowSize'] / 2) + 1] = fieldPredImg
            self._putInSharedMemory(cv.cvtColor(tempMask, cv.COLOR_GRAY2RGB), 'классификацию точек')

        if self._showStep:
            plt.imshow(fieldPredImg, cmap='gray')
            plt.show()

        # cv.imshow('EM', field_pr_img)

        # определение какому классу соответствуют значения, принадлежащие классу с минимальным математическим ожиданием
        stratumClass = np.argmin(means)
        intensiveStratum = int(stratumClass / 2 * 255)

        # выделение областей, принадлежащих классу с минимальным математическим ожиданием
        # получаем маску
        maskStratum = self._segmentForDistribution(fieldPredImg, intensiveStratum)

        # переносим получившуюся маску поля на маску для изображения
        maskStratum = cv.resize(maskStratum,
                                np.flip((self._analyzedImg.img.shape[0] - self._settings['WindowProcessing'][
                                    'windowSize'] + 1,
                                         self._analyzedImg.img.shape[1] - self._settings['WindowProcessing'][
                                             'windowSize'] + 1)),
                                fx=self._settings['WindowProcessing']['X-axisStep'],
                                fy=self._settings['WindowProcessing']['Y-axisStep'],
                                interpolation=cv.INTER_LINEAR_EXACT)
        self._analyzedImg.mask[int(self._settings['WindowProcessing']['windowSize'] / 2): -int(
            self._settings['WindowProcessing']['windowSize'] / 2) + 1,
        int(self._settings['WindowProcessing']['windowSize'] / 2): -int(
            self._settings['WindowProcessing']['windowSize'] / 2) + 1] = maskStratum

    def _maskOnImage(self, mask:np.ndarray):
        """
        Наложение маски на исходное изображение
        :param mask: ndarray
            Маска - бинарное изображение
        :return: result: ndarray
            Исходное изображение в пространстве RGB с наложенной маской
        """
        img = cv.cvtColor(self._analyzedImg.img, cv.COLOR_BGR2RGB)
        # создание слоя-заливки
        color_layer = np.zeros(img.shape, dtype=np.uint8)
        color_layer[:] = (255, 0, 102)

        # наложение заливки на изображение через маску
        # получаются закрашенные сегменты переходного слоя
        stratum = cv.bitwise_and(img, color_layer, mask=mask)

        # соединение исходного изображения с полупрозрачными закрашенными сегментами
        result = cv.addWeighted(img, 1, stratum, 0.2, 0.0)

        # выделение контуров сегментов
        color_contours = (102, 0, 153)
        contours, hierarchy = cv.findContours(mask, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
        cv.drawContours(result, contours, -1, color_contours, 1, cv.LINE_AA, hierarchy, 3)

        # перевод результата в пространство RGB
        result = cv.cvtColor(result, cv.COLOR_BGR2RGB)
        return result

    def _maskOnImageWithNumber(self, contours):
        """
        Наложение маски на исходное изображение с указанием номеров объектов
        :param contours: ndarray
            Контуры изображений
        :return: result: ndarray
            Исходное изображение в пространстве RGB с наложенной маской
        """
        imgRGB = cv.cvtColor(self._analyzedImg.img, cv.COLOR_GRAY2RGB)
        result = imgRGB
        for i, contour in enumerate(contours):
            mask = np.zeros(imgRGB.shape, dtype=np.uint8)
            mask = cv.fillPoly(mask, [contour], (255, 255, 255))
            mask = cv.cvtColor(mask, cv.COLOR_RGB2GRAY)

            # создание слоя-заливки
            color_layer = np.zeros(imgRGB.shape, dtype=np.uint8)
            color_layer[:] = self.Colors[i]

            # наложение заливки на изображение через маску
            # получаются закрашенные сегменты переходного слоя
            stratum = cv.bitwise_and(imgRGB, color_layer, mask=mask)
            # соединение исходного изображения с полупрозрачными закрашенными сегментами
            tempResult = cv.addWeighted(result, 1, stratum, 0.2, 0.0)
            # выделение контуров сегментов
            cv.drawContours(tempResult, [contour], -1, self.Colors[i], 2, cv.LINE_AA)
            radius = 15
            centre = contour[int(len(contour) / 2)][0]
            tempResult = cv.circle(tempResult, centre, radius, self.Colors[i], -1)
            tempResult = cv.putText(tempResult, str(i + 1),
                                    (centre[0] - radius + int(radius / 3), centre[1] + radius - int(radius / 3)),
                                    cv.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
            # перевод результата в пространство RGB
            result = cv.cvtColor(tempResult, cv.COLOR_BGR2RGB)

        return result

    def _segmentTransitionLayer(self):
        """
        Сегментация переходного слоя на изображении
        :return: None
        """
        if self._analyzedImg.mask is None:
            self._logging('Шаг 2/8: Определение количества компонент в смеси')
            self._analyzedImg.mask = np.zeros(self._analyzedImg.img.shape, dtype=np.uint8)
            if self._settings['WindowProcessing']['nComp'] is None:
                countComponentsObj = CountComponents(2, 3, self._showStep)
                countComponents = countComponentsObj(self._analyzedImg.field.flatten())
            else:
                countComponents = self._settings['WindowProcessing']['nComp']
            if countComponents == 2:
                self._analysisFieldWithTwoComponents()
            else:
                self._analysisFieldWithThreeComponents()
        return self._analyzedImg.mask

    def estimateTransitionLayer(self):
        """
        Оценка ширины переходного слоя
        :return: distances: list
            Значения ширины каждой сегментированной области в каждой точке их контура
        """
        self._logging('Началась оценка ширины переходного слоя')

        # Захват мутекса и общей памяти
        if self._settings['Multithreading']['Mutex'] is not None and self._settings['Multithreading']['NameOfSharedMemory'] is not None:
            print(f'{threading.current_thread().name} ожидает доступ к мутексу для первого подключения')
            self._settings['Multithreading']['Mutex'].acquire()
            print(f'{threading.current_thread().name} получил доступ к мутексу для первого подключения')
            self._sharedMemory = SharedMemory(name=self._settings['Multithreading']['NameOfSharedMemory'], create=False)
            print(f'{threading.current_thread().name} Подключение к общей памяти прошло успешно')
            self._sharedMemoryBuffer = np.ndarray(cv.cvtColor(self._analyzedImg.img, cv.COLOR_GRAY2RGB).shape,
                                                  dtype=np.uint8, buffer=self._sharedMemory.buf)

        if self._analyzedImg is None:
            raise AttributeError("It is impossible to estimate width of transition layer in absence of image")
        # Вычисление ПФР
        if self._analyzedImg.field is None and self._analyzedImg.mask is None:
            self._calculateField()
        # Сегментация переходного слоя
        if self._analyzedImg.mask is None:
            self._segmentTransitionLayer()

        if self._showStep:
            plt.imshow(self._analyzedImg.mask, cmap='gray')
            plt.show()

        self._logging('Шаг 8/8: Оценка ширины выделенных сегментов')
        # Измерение ширины каждого сегментированного объекта
        measurer = MeasureObjects(self._showStep)
        distances, contours = measurer(self._analyzedImg.mask, self._settings['WindowProcessing']['windowSize'])

        if self._settings['Multithreading']['Mutex'] is not None:
            self._putInSharedMemory(self._maskOnImageWithNumber(contours), 'нумерация объектов')
        # Освобождение мутекса и очистка общей памяти
        if self._settings['Multithreading']['Mutex'] is not None:
            self._sharedMemory.close()
            self._sharedMemory.unlink()
            print(f'{threading.current_thread().name} закрыл общую память')
            self._settings['Multithreading']['Mutex'].release()
            self._settings['Multithreading']['Mutex'] = None
            print(f'{threading.current_thread().name} освободил мутекс')

        if self._showStep:
            plt.hist(distances)
            plt.show()

        return distances
