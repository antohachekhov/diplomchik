import cv2 as cv
import matplotlib.pyplot as plt
import numpy as np
from FunctionsDistance import Line
from math import fabs, sqrt


class MeasureObjects:
    """
    Класс измерения объектов на бинарном изображении
    """
    DefaultSettings = {
        'differentiationStep': 3,
        'valueNegativeInfinity': 1e-7,
        'valuePositiveInfinity': 1e+20,
        'valueZero': 1e-4
    }

    def __init__(self, showStep: bool = False, **kwargs):
        """
        Конструктор класса
        """
        self._showStep = showStep
        self._settings = MeasureObjects.DefaultSettings
        self.setSettings(**kwargs)

        self.notFound = []

    def setSettings(self, **kwargs):
        intersectionSettings = set(**kwargs).intersection(self._settings)
        self._settings.update((keySetting, kwargs[keySetting]) for keySetting in intersectionSettings)

    @staticmethod
    def _distance(point1, point2):
        """
        Вычисление расстояния между двумя точками
        :param point1: тип list - (x,y)
        :param point2: тип list - (x,y)
        :return:
        """
        return sqrt((point1[0] - point2[0]) ** 2 + (point1[1] - point2[1]) ** 2)

    def __call__(self, mask: np.ndarray, borderSize: int = 0):
        """
        Измерение объектов на бинарном изображении
        """
        contours, hierarchy = cv.findContours(mask, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_NONE)

        measurementsObjects = []

        # цикл по каждому объекту
        for contourObject, hierarchyObject in zip(contours, hierarchy[0]):
            if hierarchyObject[-1] == -1:
                measurementsObjects.append(self._measureObject(contourObject, mask.shape, borderSize))

        return measurementsObjects, contours

    @staticmethod
    def _differentiation(data, center, left, right):
        """
        Разность между точками контура
        """
        return data[center + right] - data[center - left]

    # def _1findIntersectionWithNormalFromPoint(self, contourObject, indexPoint, derivative: float = None):
    #     [x0, y0] = contourObject[indexPoint]
    #     pointsIntersection = []
    #
    #     # область определения нормали
    #     Dx = [0, 0]
    #     Dy = [0, 0]
    #
    #     # вычисление производной
    #     if derivative is None:
    #         [dx, dy] = self._differentiation(contourObject, indexPoint,
    #                                          self._settings['differentiationStep'],
    #                                          self._settings['differentiationStep'])
    #     else:
    #         [[dx, dy]] = derivative
    #
    #     if fabs(dy) < self._settings['valueNegativeInfinity']:
    #         # for point in contourObject[np.logical_and(contourObject[:, 0] == x0,
    #         #                                               contourObject[:, 1] != y0)]:
    #         #     pointsIntersection.append(point)
    #         pointsIntersection += list(
    #             contourObject[np.logical_and(contourObject[:, 0] == x0, contourObject[:, 1] != y0)])
    #     else:
    #         if fabs(dx) < self._settings['valueNegativeInfinity']:
    #             kNorm = 0.
    #             zNorm = y0
    #             norm = Line(a=kNorm, c=zNorm)
    #
    #             # определение направления нормали
    #             testLeftPoint = cv.pointPolygonTest(contourObject, (float(x0 - 2), norm(x0 - 2)),
    #                                                 False)
    #             if cv.pointPolygonTest(contourObject, (float(x0 + 2), norm(x0 + 2)), False) >= 0:
    #                 if testLeftPoint > 0:
    #                     raise Exception("It is impossible to determine direction of normal")
    #                 else:
    #                     Dx = [x0, np.max(contourObject[:, 0])]
    #                     Dy = [y0, y0]
    #             else:
    #                 if testLeftPoint > 0:
    #                     Dx = [np.min(contourObject[:, 0]), x0]
    #                     Dy = [y0, y0]
    #                 else:
    #                     raise Exception("It is impossible to determine direction of normal")
    #         else:
    #             diffInPoint = dy / dx
    #             kNorm = -1. / diffInPoint
    #             zNorm = y0 + x0 / diffInPoint
    #             norm = Line(a=kNorm, c=zNorm)
    #
    #             noSolver = False
    #             if kNorm > 0:
    #                 testLeftPoint = cv.pointPolygonTest(contourObject,
    #                                                     (float(x0 - 2), norm(x0 - 2)), False)
    #                 if cv.pointPolygonTest(contourObject, (float(x0 + 2), norm(x0 + 2)),
    #                                        False) >= 0:
    #                     if testLeftPoint >= 0:
    #                         raise Exception("It is impossible to determine direction of normal")
    #                     else:
    #                         Dx = [x0, np.max(contourObject[:, 0])]
    #                         Dy = [y0, np.max(contourObject[:, 1])]
    #                 else:
    #                     if testLeftPoint >= 0:
    #                         Dx = [np.min(contourObject[:, 0]), x0]
    #                         Dy = [np.min(contourObject[:, 1]), y0]
    #                         AreaFounded = True
    #                     else:
    #                         raise Exception("It is impossible to determine direction of normal")
    #             else:
    #                 testLeftPoint = cv.pointPolygonTest(contourObject,
    #                                                     (float(x0 - 2), norm(x0 - 2)), False)
    #                 if cv.pointPolygonTest(contourObject, (float(x0 + 2), norm(x0 + 2)),
    #                                        False) >= 0:
    #                     if testLeftPoint >= 0:
    #                         raise Exception("It is impossible to determine direction of normal")
    #                     else:
    #                         Dx = [x0, np.max(contourObject[:, 0])]
    #                         Dy = [np.min(contourObject[:, 1]), y0]
    #                 else:
    #                     if testLeftPoint >= 0:
    #                         Dx = [np.min(contourObject[:, 0]), x0]
    #                         Dy = [y0, np.max(contourObject[:, 1])]
    #                     else:
    #                         raise Exception("It is impossible to determine direction of normal")
    #         IndexPointsInDx = np.logical_and(contourObject[:, 0] >= Dx[0],
    #                                          contourObject[:, 0] <= Dx[1])
    #         IndexPointsInDy = np.logical_and(contourObject[:, 1] >= Dy[0],
    #                                          contourObject[:, 1] <= Dy[1])
    #         PointsInD = contourObject[np.logical_and(IndexPointsInDx, IndexPointsInDy)]
    #
    #         if PointsInD.shape[0] == 0:
    #             raise Exception("No intersection with normal was found")
    #
    #         eq = norm.generalEquation(PointsInD)
    #         eq[np.abs(eq) < self._settings['valueNegativeInfinity']] = 0.
    #         # Создание массива знаков
    #         eqSign = np.sign(eq)
    #
    #         if np.all(eqSign == eqSign[0]):
    #             # Если знак не меняется, то пересечения нет
    #             raise Exception("No intersection with normal was found")
    #
    #         # Нахождение всех мест изменения знака в массиве
    #         shiftEqSign = np.roll(eqSign, -1)
    #         shiftEqSign[-1] = eqSign[-1]
    #         eqSignChange = ((shiftEqSign - eqSign) != 0).astype(int)
    #         changeIndex = np.where(eqSignChange != 0)[0]
    #         if changeIndex.shape[0] > 1:
    #             positions = [0, -1]
    #         else:
    #             positions = [0]
    #         for position in positions:
    #             if eqSign[changeIndex[position]] == 0:
    #                 # Если первое изменение знака на ноль, то соответсвующая точка - точка пересечения
    #                 pointsIntersection.append(PointsInD[changeIndex[position]])
    #             else:
    #                 # Находим точки, в которых меняется знак, в соответствии с направлением поиска
    #                 [ax, ay] = PointsInD[changeIndex[position] - 1]
    #                 [bx, by] = PointsInD[changeIndex[position]]
    #
    #                 if fabs(ax - bx) < self._settings['valueNegativeInfinity']:
    #                     xPoint = ax
    #                     yPoint = norm(xPoint)
    #                 elif fabs(ay - by) < self._settings['valueNegativeInfinity']:
    #                     yPoint = ay
    #                     xPoint = (yPoint - zNorm) / kNorm
    #                 else:
    #                     # Вычисление коэффициентов для общего уравнения кривой
    #                     k_segment = (by - ay) / (bx - ax)
    #                     z_segment = (by - ay) * (-ax / (bx - ax) + ay / (by - ay))
    #                     # Нахождение точки пересечения
    #                     xPoint = (z_segment - zNorm) / (kNorm - k_segment)
    #                     yPoint = (kNorm * z_segment - k_segment * zNorm) / (kNorm - k_segment)
    #
    #                     # if 400 < yPoint < 700 and 350 < xPoint < 650:
    #                     #     # self.interpole.append([xPoint, yPoint])
    #                     #     # self.interpolefrom.append([x0, y0])
    #                     #     self.a.append([ax, ay])
    #                     #     self.b.append([bx, by])
    #                 pointsIntersection.append(np.array([xPoint, yPoint]))
    #
    #     minPoints = None
    #     minDistance = 1e+20
    #     for point in pointsIntersection:
    #         dInPoint = self._distance(np.array([x0, y0]), point)
    #         if dInPoint < minDistance:
    #             minDistance = dInPoint
    #             minPoints = point
    #
    #     # if minDistance < 500:
    #     #     self.strange.append(minPoints)
    #
    #     if minDistance < 1e+20:
    #         return minDistance

    def _findIntersectionWithNormalFromPoint(self, contourObject, indexPoint, derivative: float = None):
        """
        Поиск точки пересечения нормали, опущенной из данной точки, с контуром объекта
        """
        [x0, y0] = contourObject[indexPoint]
        pointsIntersection = []

        # область определения нормали
        Dx = [0, 0]
        Dy = [0, 0]

        # вычисление производной
        if derivative is None:
            [dx, dy] = self._differentiation(contourObject, indexPoint,
                                             self._settings['differentiationStep'],
                                             self._settings['differentiationStep'])
        else:
            [[dx, dy]] = derivative

        if fabs(dy) < self._settings['valueNegativeInfinity']:
            # for point in contourObject[np.logical_and(contourObject[:, 0] == x0,
            #                                               contourObject[:, 1] != y0)]:
            #     pointsIntersection.append(point)
            pointsIntersection += list(
                contourObject[np.logical_and(contourObject[:, 0] == x0, contourObject[:, 1] != y0)])
        else:
            if fabs(dx) < self._settings['valueNegativeInfinity']:
                kNorm = 0.
                zNorm = y0
                norm = Line(a=kNorm, c=zNorm)

                # определение направления нормали
                testLeftPoint = cv.pointPolygonTest(contourObject, (float(x0 - 2), norm(x0 - 2)),
                                                    False)
                if cv.pointPolygonTest(contourObject, (float(x0 + 2), norm(x0 + 2)), False) >= 0:
                    if testLeftPoint > 0:
                        raise Exception("It is impossible to determine direction of normal")
                    else:
                        Dx = [x0, np.max(contourObject[:, 0])]
                        Dy = [y0, y0]
                else:
                    if testLeftPoint > 0:
                        Dx = [np.min(contourObject[:, 0]), x0]
                        Dy = [y0, y0]
                    else:
                        raise Exception("It is impossible to determine direction of normal")
                # IndexPointsInDx = np.logical_and(contourObject[:, 0] >= Dx[0],
                #                                  contourObject[:, 0] <= Dx[1])
                # IndexPointsInDy = np.logical_and(contourObject[:, 1] >= Dy[0],
                #                                  contourObject[:, 1] <= Dy[1])
            else:
                diffInPoint = dy / dx
                kNorm = -1. / diffInPoint
                zNorm = y0 + x0 / diffInPoint
                norm = Line(a=kNorm, c=zNorm)

                noSolver = False
                if kNorm > 0:
                    testLeftPoint = cv.pointPolygonTest(contourObject,
                                                        (float(x0 - 2), norm(x0 - 2)), False)
                    if cv.pointPolygonTest(contourObject, (float(x0 + 2), norm(x0 + 2)),
                                           False) >= 0:
                        if testLeftPoint >= 0:
                            raise Exception("It is impossible to determine direction of normal")
                        else:
                            Dx = [x0 + 1, np.max(contourObject[:, 0])]
                            Dy = [y0 + 1, np.max(contourObject[:, 1])]
                    else:
                        if testLeftPoint >= 0:
                            Dx = [np.min(contourObject[:, 0]), x0 - 1]
                            Dy = [np.min(contourObject[:, 1]), y0 - 1]
                            AreaFounded = True
                        else:
                            raise Exception("It is impossible to determine direction of normal")
                else:
                    testLeftPoint = cv.pointPolygonTest(contourObject,
                                                        (float(x0 - 2), norm(x0 - 2)), False)
                    if cv.pointPolygonTest(contourObject, (float(x0 + 2), norm(x0 + 2)),
                                           False) >= 0:
                        if testLeftPoint >= 0:
                            raise Exception("It is impossible to determine direction of normal")
                        else:
                            Dx = [x0 + 1, np.max(contourObject[:, 0])]
                            Dy = [np.min(contourObject[:, 1]), y0 - 1]
                    else:
                        if testLeftPoint >= 0:
                            Dx = [np.min(contourObject[:, 0]), x0 - 1]
                            Dy = [y0 + 1, np.max(contourObject[:, 1])]
                        else:
                            raise Exception("It is impossible to determine direction of normal")

            PointsInD = np.roll(contourObject, -np.where(np.logical_and(contourObject[:, 0] == x0, contourObject[:, 1] == y0))[0][0], axis=0)
            IndexPointsInDx = np.logical_and(PointsInD[:, 0] >= Dx[0],
                                             PointsInD[:, 0] <= Dx[1])
            IndexPointsInDy = np.logical_and(PointsInD[:, 1] >= Dy[0],
                                             PointsInD[:, 1] <= Dy[1])
            PointsInD = PointsInD[np.logical_and(IndexPointsInDx, IndexPointsInDy)]

            PointsInD = PointsInD[np.logical_or(PointsInD[:, 0] != x0, PointsInD[:, 1] != y0)]

            if PointsInD.shape[0] == 0:
                raise Exception(
                    f"No intersection with normal was found. No one point in area D. dy={dy}, dx={dx}>{self._settings['valueNegativeInfinity']}")

            eq = norm.generalEquation(PointsInD)
            eq[np.abs(eq) < self._settings['valueNegativeInfinity']] = 0.
            # Создание массива знаков
            eqSign = np.sign(eq)

            if eqSign.shape[0] == 1 and eqSign[0] == 0:
                pointsIntersection.append(PointsInD[0])
                # print(f'Найдена только одна точка пересечения в области dx={dx} dy={dy} длина {self._distance(np.array([x0, y0]), PointsInD[0])}')
                # self.onlyOne.append(PointsInD[0])
            else:
                if np.all(eqSign == eqSign[0]):
                    # Если знак не меняется, то пересечения нет
                    raise Exception(f"No intersection with normal was found. Sign not change")

                # Нахождение всех мест изменения знака в массиве
                shiftEqSign = np.roll(eqSign, -1)
                shiftEqSign[-1] = eqSign[-1]
                eqSignChange = ((shiftEqSign - eqSign) != 0).astype(int)
                changeIndex = np.where(eqSignChange != 0)[0]
                if changeIndex.shape[0] > 1:
                    positions = [0, -1]
                else:
                    positions = [0]
                for position in positions:
                    if eqSign[changeIndex[position]] == 0:
                        # Если первое изменение знака на ноль, то соответсвующая точка - точка пересечения
                        pointsIntersection.append(PointsInD[changeIndex[position]])
                    else:
                        # Находим точки, в которых меняется знак, в соответствии с направлением поиска
                        [ax, ay] = PointsInD[changeIndex[position] - 1]
                        [bx, by] = PointsInD[changeIndex[position]]

                        if fabs(ax - bx) < self._settings['valueNegativeInfinity']:
                            xPoint = ax
                            yPoint = norm(xPoint)

                        elif fabs(ay - by) < self._settings['valueNegativeInfinity']:
                            yPoint = ay
                            xPoint = (yPoint - zNorm) / kNorm
                        else:
                            # Вычисление коэффициентов для общего уравнения кривой
                            k_segment = (by - ay) / (bx - ax)
                            z_segment = (by - ay) * (-ax / (bx - ax) + ay / (by - ay))
                            # Нахождение точки пересечения
                            xPoint = (z_segment - zNorm) / (kNorm - k_segment)
                            yPoint = (kNorm * z_segment - k_segment * zNorm) / (kNorm - k_segment)

                            # print([x0, y0] == [xPoint, yPoint])
                            # if 400 < yPoint < 700 and 350 < xPoint < 650:
                            #     self.interpole.append([xPoint, yPoint])
                            #     self.interpolefrom.append([x0, y0])
                            #     self.a.append([ax, ay])
                            #     self.b.append([bx, by])
                        # if [x0, y0] == [xPoint, yPoint]:
                        #     pass
                        if [x0, y0] != [xPoint, yPoint]:
                            pointsIntersection.append(np.array([xPoint, yPoint]))

        minPoints = None
        minDistance = 1e+20
        for point in pointsIntersection:
            dInPoint = self._distance(np.array([x0, y0]), point)
            if dInPoint < minDistance:
                minDistance = dInPoint
                minPoints = point

        # if minDistance < 100:
        #     self.strange.append(minPoints)
        #     self.strangefrom.append([x0, y0])

        if minDistance < 1e+20:
            return minDistance

    def _defineAreaDifferentiation(self, contourObject, indexPoint, searchArea) -> tuple[int, int]:
        """
        Определение области вычисления производной
        """
        globalIndexesContourObjectFiltered = np.where(searchArea)[0]
        # поиск точек разрыва
        left = 0
        right = 0
        borderFounded = False
        while not borderFounded and right < self._settings['differentiationStep']:
            indexing = indexPoint + right
            if indexing >= contourObject.shape[0]:
                indexing -= contourObject.shape[0]
            indexingAfter = indexPoint + right + 1
            if indexingAfter >= contourObject.shape[0]:
                indexingAfter -= contourObject.shape[0]
            if globalIndexesContourObjectFiltered[indexingAfter] - \
                    globalIndexesContourObjectFiltered[indexing] == 1:
                right += 1
            else:
                borderFounded = True
        while not borderFounded and left < self._settings['differentiationStep']:
            indexing = indexPoint - left
            if indexing < 0:
                indexing = contourObject.shape[0] + indexPoint - left
            indexingAfter = indexPoint - left - 1
            if indexingAfter >= contourObject.shape[0]:
                indexingAfter = contourObject.shape[0] + indexPoint - left
            if globalIndexesContourObjectFiltered[indexing] - \
                    globalIndexesContourObjectFiltered[indexingAfter] == 1:
                left += 1
            else:
                borderFounded = True

        if left == self._settings['differentiationStep'] and right < self._settings['differentiationStep']:
            right = 0
        if right == self._settings['differentiationStep'] and left < self._settings['differentiationStep']:
            left = 0
        if left == right == 0:
            return -1, 1
        else:
            return left, right

    def _measureObject(self, contourObject: np.ndarray, shape, borderSize: int = 0):
        """
        Измерение заданного объекта
        """
        distances = []

        # составление множества точек, не входящих в зону рамки маски
        areaX = np.logical_and(contourObject[:, :, 0] > borderSize / 2,
                               contourObject[:, :, 0] < shape[1] - borderSize / 2)
        areaY = np.logical_and(contourObject[:, :, 1] > borderSize / 2,
                               contourObject[:, :, 1] < shape[0] - borderSize / 2)
        area = np.logical_and(areaX, areaY)
        contourObjectFiltered = contourObject[area]

        # проход по каждой точке контура
        for indexPoint, _ in enumerate(contourObjectFiltered):
            try:
                left, right = self._defineAreaDifferentiation(contourObject, indexPoint, area)
                if left == -1:
                    raise Exception("It is impossible to construct a normal")
                else:
                    distance = self._findIntersectionWithNormalFromPoint(contourObjectFiltered,
                                                                         indexPoint,
                                                                         derivative=self._differentiation(contourObject,
                                                                                                          indexPoint,
                                                                                                          left, right))
                    if distance is not None:
                        distances.append(distance)
                    else:
                        self.notFound.append(contourObjectFiltered[indexPoint])
            except Exception as e:
                # print(e)
                self.notFound.append(contourObjectFiltered[indexPoint])
                continue
        return distances
