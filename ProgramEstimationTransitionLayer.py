import time
import io
import numpy as np
from WindowProcessing import WindowProcessing
import FractalAnalysisInImage
from CountComponents import CountComponents
from scipy.signal import medfilt
from sklearn.neighbors import NearestNeighbors
from KneeLocatorModule import KneeLocator
from sklearn.cluster import DBSCAN
import cv2 as cv
import matplotlib.pyplot as plt
from statsmodels.nonparametric.api import lowess
from sklearn.mixture import GaussianMixture
from MeasurementOnBinaryImage import MeasureObjects
from Image import Image
from PIL import Image as PILImage
from multiprocessing.shared_memory import SharedMemory

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
            'widthOfMedianKernel': 21,
            'columnarAnalysis': True,
            'LOWESS-frac': 0.1
        },
        'Measure': {
            'CentralTendency': 'median'
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
        -1: (139, 0, 139), # темно-пурпурный
        0: (0, 255, 255), # желтый
        1: (180, 119, 31),  # голубой
        2: tuple(reversed((44, 160, 44))), # зелёный
        3: tuple(reversed((214, 39, 40))), # красный
        4: tuple(reversed((148, 103, 189))), # пурпурный
        5: tuple(reversed((140, 86, 75))), # коричневый
        6: tuple(reversed((227, 119, 194))), # розовый
        7: tuple(reversed((188, 189, 34))), # оливковый
        8: tuple(reversed((23, 190, 207))), # циан
        9: (0, 0, 128), # бордовый
        10: (80, 127, 255), # коралл
        11: (0, 165, 255), # апельсин
        12: (212, 255, 127), # Аквамарин
        13: (0, 252, 124), # газон зеленый
        14: (114, 128, 250), # лосось
        15: (128, 0, 0), # флот
        16: (50, 205, 154), # желтый зеленый
        17: (180, 105, 255), # ярко-розовый
        18: (130, 0, 75), # индиго
        19: (255, 0, 0), # синий
        20: (14, 127, 255),  # оранжевый
        21: (50, 205, 50), # зеленый лайм
        22: (140, 230, 240), # хаки
        23: (34, 139, 34), # зеленый лес
        24: (154, 250, 0), # средний весенний
        25: (0, 215, 255), # золото
        26: (237, 149, 100), # кукуруза цветок синий
        27: (144, 238, 144), # светло-зеленый
        28: (255, 255, 0), # аква
        29: (113, 179, 60), # средний морской зеленый
    }

    def __init__(self, showStep:bool=False, mutex=None, nameSharedMemory=None):
        self._analyzedImg = None
        self._showStep = showStep
        self._settings = EstimationTransitionLayer.DefaultSettings
        self._mutex = mutex
        self._sharedMemory = None
        self._sharedMemoryBuffer = None
        if mutex is not None and nameSharedMemory is not None:
            self._mutex.acquire()
            self._sharedMemory = SharedMemory(name=nameSharedMemory, create=False)

    def clear(self):
        self._analyzedImg = None
        self._settings = EstimationTransitionLayer.DefaultSettings
        self._sharedMemory = None
        if self._mutex is not None:
            self._mutex.release()

    def setSettings(self, **kwargs):
        for key in kwargs:
            inter = set(kwargs[key]).intersection(self._settings[key])
            self._settings[key].update((keyIntersec, kwargs[key][keyIntersec]) for keyIntersec in inter)

    def _checkField(self, image, field:np.ndarray) -> bool:
        # размер поля должен соответсвовать размеру изображения
        if field.shape != (int((image.shape[0] - self._settings['WindowProcessing']['windowSize']) /
                               self._settings['WindowProcessing']['Y-axisStep'] + 1),
                           int((image.shape[1] - self._settings['WindowProcessing']['windowSize']) /
                               self._settings['WindowProcessing']['X-axisStep'] + 1)):
            raise ValueError('The size of the field does not correspond to the size of the image and'
                             ' the selected window parameters')
        return True

    def setImage(self, image:np.ndarray, field:np.ndarray=None, mask:np.ndarray=None):
        if field is not None:
            self._checkField(image, field)
        self._analyzedImg = Image(image, field, mask)

        if self._sharedMemory is not None:
            self._sharedMemoryBuffer = np.ndarray(cv.cvtColor(self._analyzedImg.img, cv.COLOR_GRAY2RGB).shape,
                                                  dtype=np.uint8, buffer=self._sharedMemory.buf)


    def getField(self):
        return self._analyzedImg.field

    def setField(self, field:np.ndarray):
        if self._analyzedImg is None:
            raise AttributeError("It is not possible to set field in the absence of image")
        self._checkField(field)
        self._analyzedImg.field = field

    def getMask(self):
        return self._analyzedImg.mask

    def setMask(self, mask:np.ndarray):
        self._analyzedImg.mask = mask

    def _calculateField(self):
        winProcess = WindowProcessing(**self._settings['WindowProcessing'])
        subWindowsSizes = np.array(
            [int((self._settings['WindowProcessing']['windowSize'] + (2 ** degree - 1)) / 2 ** degree)
             for degree in range(self._settings['WindowProcessing']['numberWindowDivides'] + 1)])
        self._analyzedImg.field = winProcess.processing(self._analyzedImg.img,
                                                        EstimationTransitionLayer.WindowFunctions[
                                                self._settings['WindowProcessing'][
                                                    'fractalDimensionEstimationMethod']],
                                                        subWindowsSizes)

    @staticmethod
    def _findExtr(data:np.ndarray):
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

    @staticmethod
    def _fig2img(fig):
        buf = io.BytesIO()
        fig.savefig(buf)
        buf.seek(0)
        img = PILImage.open(buf)
        return img

    def _analysisFieldWithTwoComponents(self):
        """
        Сегментация переходного слоя по сильным изменения поля

        """
        # массивы точек начала и конца сильных изменений соответственно
        changeBegin = np.array([0, 0], dtype=np.uint32)
        changeEnd = np.array([0, 0], dtype=np.uint32)

        # проход по каждому столбцу в поле
        for n, sliceOfField in enumerate(self._analyzedImg.field.T if self._settings['AnalysisOfFieldWithTwoComponents']['columnarAnalysis'] else self._analyzedImg.field):
            # медианная фильтрация
            filterSlice = medfilt(sliceOfField,
                                  kernel_size=self._settings['AnalysisOfFieldWithTwoComponents']['widthOfMedianKernel'])

            # координаты локальных экстремумов и разность значений между ними
            amplMF, extrMF = self._findExtr(filterSlice)

            # Выделение областей, в которых разность между экстремумами больше,
            # чем величина изменения фрактальной размерности на границе двух текстур
            indexPeak = np.where(amplMF > self._settings['AnalysisOfFieldWithTwoComponents']['minChangeFractalDimensionOfTwoTextures'])
            if indexPeak[0].size > 0:
                for j in indexPeak[0]:
                    # области на границе изображения не берутся во внимание,
                    # так как на границах поле искажено
                    if j != 0 and j < extrMF.shape[0] - 1:
                        if 0 < extrMF[j] < self._analyzedImg.field.shape[self._settings['AnalysisOfFieldWithTwoComponents']['columnarAnalysis']] - 1 and extrMF[j + 1] < self._analyzedImg.field.shape[self._settings['AnalysisOfFieldWithTwoComponents']['columnarAnalysis']] - 1:
                            changeBegin = np.vstack([changeBegin, np.array([n, extrMF[j]])])
                            changeEnd = np.vstack([changeEnd, np.array([n, extrMF[j + 1]])])

        changeBegin = np.delete(changeBegin, 0, axis=0)
        changeEnd = np.delete(changeEnd, 0, axis=0)

        # перевод точек поля в размерность изображения
        for point in changeBegin:
            point[0] = int(point[0] * self._settings['WindowProcessing']['X-axisStep'] + self._settings['WindowProcessing']['windowSize'] / 2)
            point[1] = int(point[1] * self._settings['WindowProcessing']['Y-axisStep'] + self._settings['WindowProcessing']['windowSize'] / 2)

        for point in changeEnd:
            point[0] = int(point[0] * self._settings['WindowProcessing']['X-axisStep'] + self._settings['WindowProcessing']['windowSize'] / 2)
            point[1] = int(point[1] * self._settings['WindowProcessing']['Y-axisStep'] + self._settings['WindowProcessing']['windowSize'] / 2)

        if self._mutex is not None:
            # plt.imshow(cv.cvtColor(self._analyzedImg.img, cv.COLOR_GRAY2RGB))
            # plt.scatter(changeBegin[:, 0], changeBegin[:, 1], c='red', s=1)
            # plt.scatter(changeEnd[:, 0], changeEnd[:, 1], c='red', s=1)
            # fig = plt.gcf()
            # self._sharedMemory = self._fig2img(fig)
            tempImg = cv.cvtColor(self._analyzedImg.img, cv.COLOR_GRAY2BGR)
            for b, e in zip(changeBegin, changeEnd):
                tempImg = cv.circle(tempImg, (b[0], b[1]), radius=3, color=(0, 0, 255), thickness=-1)
                tempImg = cv.circle(tempImg, (e[0], e[1]), radius=3, color=(0, 0, 255), thickness=-1)
            self._sharedMemoryBuffer[:] = cv.cvtColor(tempImg, cv.COLOR_BGR2RGB)

            self._mutex.release()
            time.sleep(1)
            self._mutex.acquire()

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
        dbscanCluster = DBSCAN(eps=self._settings['DBSCAN']['maxDistanceBetweenPointsInSingleCluster'], min_samples=self._settings['DBSCAN']['minSamplesInCluster'])
        dbscanCluster.fit(changeBegin)

        if self._mutex is not None:
            # plt.imshow(cv.cvtColor(self._analyzedImg.img, cv.COLOR_GRAY2RGB))
            # plt.scatter(changeBegin[:, 0], changeBegin[:, 1], c=dbscanCluster.labels_, s=15)
            # plt.scatter(changeEnd[:, 0], changeEnd[:, 1], c=dbscanCluster.labels_, s=15)
            # fig = plt.gcf()
            # self._sharedMemory = self._fig2img(fig)

            tempImg = cv.cvtColor(self._analyzedImg.img, cv.COLOR_GRAY2BGR)
            for i, z in enumerate(zip(changeBegin, changeEnd)):
                b, e = z
                tempImg = cv.circle(tempImg, (b[0], b[1]), radius=3,
                                    color=EstimationTransitionLayer.Colors[dbscanCluster.labels_[i]], thickness=-1)
                tempImg = cv.circle(tempImg, (e[0], e[1]), radius=3,
                                    color=EstimationTransitionLayer.Colors[dbscanCluster.labels_[i]], thickness=-1)
            self._sharedMemoryBuffer[:] = cv.cvtColor(tempImg, cv.COLOR_BGR2RGB)

            self._mutex.release()
            time.sleep(1)
            self._mutex.acquire()

        # получение списка номеров кластеров и их размеров
        uniqLabels, labelCount = np.unique(dbscanCluster.labels_, return_counts=True)

        # средний размер кластеров, не считая кластер шума
        labelCountMean = np.mean(labelCount[1:])

        # массивы для точек из не отброшенных кластеров
        clusteredChangeBegin = np.array([0, 0], dtype=np.uint32)
        clusteredChangeEnd = np.array([0, 0], dtype=np.uint32)
        filteredLabels = np.empty(0, dtype=np.uint8)

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

        if self._mutex is not None:
            # plt.imshow(cv.cvtColor(self._analyzedImg.img, cv.COLOR_GRAY2RGB))
            # plt.scatter(clusteredChangeBegin[:, 0], clusteredChangeBegin[:, 1], c=filteredLabels, s=10)
            # plt.scatter(clusteredChangeEnd[:, 0], clusteredChangeEnd[:, 1], c=filteredLabels, s=10)
            # fig = plt.gcf()
            # self._sharedMemory = self._fig2img(fig)

            tempImg = cv.cvtColor(self._analyzedImg.img, cv.COLOR_GRAY2BGR)
            for i, z in enumerate(zip(clusteredChangeBegin, clusteredChangeEnd)):
                b, e = z
                tempImg = cv.circle(tempImg, (b[0], b[1]), radius=3,
                                    color=EstimationTransitionLayer.Colors[filteredLabels[i]], thickness=-1)
                tempImg = cv.circle(tempImg, (e[0], e[1]), radius=3,
                                    color=EstimationTransitionLayer.Colors[filteredLabels[i]], thickness=-1)
            self._sharedMemoryBuffer[:] = cv.cvtColor(tempImg, cv.COLOR_BGR2RGB)

            self._mutex.release()
            time.sleep(1)
            self._mutex.acquire()

        if self._showStep:
            # перевод изображения в цветовое пространство RGB
            imgRGB = cv.cvtColor(self._analyzedImg.img, cv.COLOR_GRAY2RGB)
            fig, axis = plt.subplots(2, 2)

            axis[0, 0].imshow(imgRGB)
            axis[0, 0].scatter(changeBegin[:, 0], changeBegin[:, 1], c='red', s=1)
            axis[0, 0].scatter(changeEnd[:, 0], changeEnd[:, 1], c='red', s=1)
            axis[0, 0].set_title("Image and amplitude's points")

            axis[0, 1].set_title('Knee Point = ' + str(self._settings['DBSCAN']['maxDistanceBetweenPointsInSingleCluster']))
            axis[0, 1].plot(knee.x, knee.y, "b", label="data")
            axis[0, 1].vlines(knee.knee, plt.ylim()[0], plt.ylim()[1], linestyles="--", label="knee/elbow")
            axis[0, 1].legend(loc="best")

            axis[1, 0].scatter(changeBegin[:, 0], changeBegin[:, 1], c=dbscanCluster.labels_, s=15)
            axis[1, 0].scatter(changeEnd[:, 0], changeEnd[:, 1], c=dbscanCluster.labels_, s=15)
            axis[1, 0].set_title('DBSCAN eps=' + str(self._settings['DBSCAN']['maxDistanceBetweenPointsInSingleCluster']) + ' min_samples=' + str(self._settings['DBSCAN']['minSamplesInCluster']))

            axis[1, 1].scatter(clusteredChangeBegin[:, 0], clusteredChangeBegin[:, 1], c=filteredLabels, s=10)
            axis[1, 1].scatter(clusteredChangeEnd[:, 0], clusteredChangeEnd[:, 1], c=filteredLabels, s=10)
            axis[1, 1].set_title('[Filtered] DBSCAN eps=' + str(self._settings['DBSCAN']['maxDistanceBetweenPointsInSingleCluster']) + ' m_samp=' + str(self._settings['DBSCAN']['minSamplesInCluster']))

            plt.show()
            plt.close()

        # if self._mutex is not None:
        #     # plt.imshow(cv.cvtColor(self._analyzedImg.img, cv.COLOR_GRAY2RGB))
        #     self._sharedMemory = cv.cvtColor(self._analyzedImg.img, cv.COLOR_GRAY2RGB)
        #
        #     self._mutex.release()
        #     time.sleep(1)
        #     self._mutex.acquire()

        self.listDistancesInEveryPoint = []

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

            if self._mutex is not None:
                # cv.drawContours(tempImg, (z1.reshape((z1.shape[0], 1, 2)), ), -1, (255, 0, 0), 3)
                # cv.drawContours(tempImg, (z2.reshape((z2.shape[0], 1, 2)), ), -1, (255, 0, 0), 3)
                cv.polylines(tempImg, [z1.astype(np.int32).reshape((-1, 1, 2))], False, (255, 0, 0), 2)
                cv.polylines(tempImg, [z2.astype(np.int32).reshape((-1, 1, 2))], False, (255, 0, 0), 2)

                # plt.plot(z1[:, 0], z1[:, 1], '--', c='blue', linewidth=3)
                # plt.plot(z2[:, 0], z2[:, 1], '--', c='blue', linewidth=3)

            # округляем точки до целых, чтобы поставить им в соответствие пиксели изображения
            pointsBeg = np.around(z1).astype(np.int32)
            pointsBeg = pointsBeg.reshape((-1, 1, 2))
            pointsEnd = np.around(z2).astype(np.int32)
            pointsEnd = np.flipud(pointsEnd).reshape((-1, 1, 2))
            pts = np.concatenate([pointsBeg, pointsEnd, pointsBeg[0].reshape((-1, 1, 2))])

            # рисуем на маске линии начала и конца изменения и закрашиваем область между ними
            cv.fillPoly(self._analyzedImg.mask, [pts], 255)


        if self._mutex is not None:
            # fig = plt.gcf()
            # self._sharedMemory = self._fig2img(fig)
            self._sharedMemoryBuffer[:] = cv.cvtColor(tempImg, cv.COLOR_BGR2RGB)
            self._mutex.release()
            time.sleep(1)
            self._mutex.acquire()



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
        maskByIntensity = cv.inRange(img, intensity, intensity + 1)

        # "открытие" изображения (удаление мелких объектов) с эллиптическим ядром 7х7
        kernel = cv.getStructuringElement(cv.MORPH_RECT, (7, 7))
        maskAfterOpen = cv.morphologyEx(maskByIntensity, cv.MORPH_OPEN, kernel)

        # "закрытие" изображения (заполнение мелких пропусков в объектах) с эллиптическим ядром 5х5
        kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE, (5, 5))
        maskAfterClose = cv.morphologyEx(maskAfterOpen, cv.MORPH_CLOSE, kernel)

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

        # наращивание объектов
        radius = int(self._settings['WindowProcessing']['windowSize'] / 2 if self._settings['WindowProcessing']['windowSize'] / 2 % 2 == 1 else self._settings['WindowProcessing']['windowSize'] / 2 - 1)
        kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE, (radius, radius))
        maskAfterDilate = cv.dilate(maskAfterFilter, kernel)

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
            plt.imshow(maskAfterDilate, cmap='gray')
            plt.title('Blur&Thresh')
            plt.xticks([])
            plt.yticks([])

            plt.show()

        return maskAfterDilate

    def _analysisFieldWithThreeComponents(self):
        """
        Сегментация переходного слоя через EM-классификацию

        :param n_comp: int
            Количество компонент в смеси распределений

        """
        # классификация значений поля по EM-алгоритму с тремя компонентами в смеси
        X = np.expand_dims(self._analyzedImg.field.flatten(), 1)
        model = GaussianMixture(n_components=3, covariance_type='full')
        model.fit(X)
        predict = model.predict(X)
        means = model.means_
        fieldPredict = np.resize(predict, self._analyzedImg.field.shape)

        # перевод поля значений классов в пространство оттенков серого
        fieldPredImg = fieldPredict / 2 * 255
        fieldPredImg = fieldPredImg.astype(np.uint8)

        # cv.imshow('EM', field_pr_img)

        # определение какому классу соответствуют значения, принадлежащие классу с минимальным математическим ожиданием
        stratumClass = np.argmin(means)
        intensiveStratum = int(stratumClass / 2 * 255)

        # выделение областей, принадлежащих классу с минимальным математическим ожиданием
        # получаем маску
        maskStratum = self._segmentForDistribution(fieldPredImg, intensiveStratum)

        # переносим получившуюся маску поля на маску для изображения
        maskStratum = cv.resize(maskStratum,
                                np.flip((self._analyzedImg.img.shape[0] - self._settings['WindowProcessing']['windowSize'] + 1,
                                         self._analyzedImg.img.shape[1] - self._settings['WindowProcessing']['windowSize'] + 1)),
                                fx=self._settings['WindowProcessing']['X-axisStep'], fy=self._settings['WindowProcessing']['Y-axisStep'], interpolation=cv.INTER_LINEAR_EXACT)
        self._analyzedImg.mask[int(self._settings['WindowProcessing']['windowSize'] / 2): -int(self._settings['WindowProcessing']['windowSize'] / 2) + 1,
        int(self._settings['WindowProcessing']['windowSize'] / 2): -int(self._settings['WindowProcessing']['windowSize'] / 2) + 1] = maskStratum

    def _segmentTransitionLayer(self):
        if self._analyzedImg.mask is None:
            self._analyzedImg.mask = np.zeros(self._analyzedImg.img.shape, dtype=np.uint8)
            countComponentsObj = CountComponents(2, 3, self._showStep)
            countComponents = countComponentsObj(self._analyzedImg.field.flatten())
            if countComponents == 2:
                self._analysisFieldWithTwoComponents()
            else:
                self._analysisFieldWithThreeComponents()
        return self._analyzedImg.mask

    def estimateTransitionLayer(self):
        if self._analyzedImg is None:
            raise AttributeError("It is impossible to estimate width of transition layer in absence of image")
        if self._analyzedImg.field is None and self._analyzedImg.mask is None:
            self._calculateField()

            # if self._mutex is not None:
            #     self._sharedMemoryBuffer[:] = cv.cvtColor(self.fieldToImage(), cv.COLOR_GRAY2RGB)
            #     self._mutex.release()
            #     time.sleep(1)
            #     self._mutex.acquire()

        if self._analyzedImg.mask is None:
            self._segmentTransitionLayer()

            if self._showStep:
                plt.imshow(self._analyzedImg.mask, cmap='gray')
                plt.show()

        if self._mutex is not None:
            self._sharedMemoryBuffer[:] = cv.cvtColor(self._analyzedImg.mask, cv.COLOR_GRAY2RGB)
            self._mutex.release()
            time.sleep(1)
            self._mutex.acquire()

        measurer = MeasureObjects(self._showStep)
        distances = measurer(self._analyzedImg.img, self._analyzedImg.mask, self._settings['WindowProcessing']['windowSize'])

        # if self._showStep:
        #     plt.hist(distances)
        #     plt.show()

        return distances

        # result = list(map(EstimationTransitionLayer.CentralTendency[self._settings['Measure']['CentralTendency']],
        #                   distances))
        #
        # return result

    def fieldToImage(self):
        """
        Перевод поля фрактальных размерностей в пространство оттенков серого

        :param field: 2-D numpy массив
            Поле фрактальных размерностей.

        :return imgOut: 2-D numpy массив
            Изображение-визуализация поля фрактальных размерностей
        """
        fieldConversionFunc = lambda x: int(255 * (x - 1) / 3) if x - 1.0 > 0.0 else 0
        fieldImg = np.vectorize(fieldConversionFunc)(self._analyzedImg.field)
        return fieldImg