import cv2 as cv
import matplotlib.pyplot as plt
import numpy as np
from FunctionsDistance import Line
from math import fabs, sqrt

class Measure:
    def __init__(self):
        pass

    @staticmethod
    def distance(point1, point2):
        """
        Вычисление расстояния между двумя точками
        :param point1: тип list - (x,y)
        :param point2: тип list - (x,y)
        :return:
        """
        return sqrt((point1[0] - point2[0]) ** 2 + (point1[1] - point2[1]) ** 2)

    def __call__(self, image, mask, winSize, *args, **kwargs):
        contours0, hierarchy = cv.findContours(mask, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)

        index = 0
        layer = 0

        def update():
            vis = image.copy()
            cv.drawContours(vis, contours0, index, (150, 0, 0), 2, cv.LINE_AA, hierarchy, layer)
            cv.imshow('contours', vis)

        def update_index(v):
            global index
            index = v - 1
            update()

        def update_layer(v):
            global layer
            layer = v
            update()

        update_index(0)
        update_layer(0)
        cv.createTrackbar("contour", "contours", 0, 7, update_index)
        cv.createTrackbar("layers", "contours", 0, 7, update_layer)

        cv.waitKey()
        cv.destroyAllWindows()

        for i in range(len(contours0)):
            if hierarchy[0, i, 3] == -1:
                plt.plot(contours0[i][:, 0, 0], contours0[i][:, 0, 1])
        plt.show()

        nDiff = 3
        eps = 1e-7
        contours = contours0
        index_object = 0
        distances = []

        for index_object in range(len(contours0)):
            contourObject = contours[index_object]
            areaX = np.logical_and(contourObject[:, :, 0] > winSize / 2,
                                   contourObject[:, :, 0] < image.shape[0] - winSize / 2)
            areaY = np.logical_and(contourObject[:, :, 1] > winSize / 2,
                                   contourObject[:, :, 1] < image.shape[1] - winSize / 2)
            area = np.logical_and(areaX, areaY)
            contourObjectFiltered = contourObject[area]
            globalIndexesContourObjectFiltered = np.where(area)[0]

            for localIndex, [x0, y0] in enumerate(contourObjectFiltered):
                try:
                    # print(localIndex)
                    # localIndex = 100
                    [x0, y0] = contourObjectFiltered[localIndex]
                    # удаление точек на границах маски
                    Points = []
                    left = False
                    up = False
                    Dx = [0, 0]
                    Dy = [0, 0]

                    il = 0
                    ir = 0
                    borderFounded = False
                    while not borderFounded and ir < nDiff:
                        indexing = localIndex + ir
                        if indexing >= contourObject.shape[0]:
                            indexing -= contourObject.shape[0]
                        indexingAfter = localIndex + ir + 1
                        if indexingAfter >= contourObject.shape[0]:
                            indexingAfter -= contourObject.shape[0]
                        if globalIndexesContourObjectFiltered[indexingAfter] - \
                                globalIndexesContourObjectFiltered[indexing] == 1:
                            # or globalIndexesContourObjectFiltered[localIndex + ir + 1] in [0, contourObject.shape[0] - 1] and \
                            # globalIndexesContourObjectFiltered[localIndex + ir] in [0, contourObject.shape[0] - 1]:
                            ir += 1
                        else:
                            borderFounded = True
                    while not borderFounded and il < nDiff:
                        indexing = localIndex - il
                        if indexing < 0:
                            indexing = contourObject.shape[0] + localIndex - il
                        indexingAfter = localIndex - il - 1
                        if indexingAfter >= contourObject.shape[0]:
                            indexingAfter = contourObject.shape[0] + localIndex - il
                        if globalIndexesContourObjectFiltered[indexing] - \
                                globalIndexesContourObjectFiltered[indexingAfter] == 1:
                            # or globalIndexesContourObjectFiltered[localIndex - il - 1] in [0, contourObject.shape[0] - 1] and \
                            # globalIndexesContourObjectFiltered[localIndex - il] in [0, contourObject.shape[0] - 1]:
                            il += 1
                        else:
                            borderFounded = True
                    if il == ir and il != 0:
                        # центральная производная
                        [dx, dy] = contourObjectFiltered[localIndex + ir] - contourObjectFiltered[localIndex - il]
                    elif il == nDiff and il < nDiff:
                        # левая производная
                        [dx, dy] = contourObjectFiltered[localIndex] - contourObjectFiltered[localIndex - il]
                    elif ir == nDiff and il < nDiff:
                        # правая производная
                        [dx, dy] = contourObjectFiltered[localIndex + ir] - contourObjectFiltered[localIndex]
                    else:
                        # построить нормаль невозможно
                        continue

                    if fabs(dy) < eps:
                        for point in contourObjectFiltered[np.logical_and(contourObjectFiltered[:, 0] == x0,
                                                                          contourObjectFiltered[:, 1] != y0)]:
                            Points.append(point)
                    else:
                        if fabs(dx) < eps:
                            k_norm = 0.
                            z_norm = y0
                            norm = Line(a=k_norm, c=z_norm)
                            testLeftPoint = cv.pointPolygonTest(contourObjectFiltered, (float(x0 - 1), norm(x0 - i)),
                                                                False)
                            if cv.pointPolygonTest(contourObjectFiltered, (float(x0 + 1), norm(x0 + i)), False) >= 0:
                                if testLeftPoint > 0:
                                    continue
                                else:
                                    left = False
                                    Dx = [x0, np.max(contourObjectFiltered[:, 0])]
                                    Dy = [y0 - 1, y0 + 1]
                            else:
                                if testLeftPoint > 0:
                                    left = True
                                    Dx = [np.min(contourObjectFiltered[:, 0]), x0]
                                    Dy = [y0, y0]
                                else:
                                    continue
                        else:
                            diffInPoint = dy / dx
                            k_norm = -1. / diffInPoint
                            z_norm = y0 + x0 / diffInPoint
                            norm = Line(a=k_norm, c=z_norm)

                            noSolver = False
                            i = 1
                            if k_norm > 0:
                                AreaFounded = False
                                while not AreaFounded:
                                    testLeftPoint = cv.pointPolygonTest(contourObjectFiltered,
                                                                        (float(x0 - i), norm(x0 - i)), False)
                                    if cv.pointPolygonTest(contourObjectFiltered, (float(x0 + i), norm(x0 + i)),
                                                           False) >= 0:
                                        if testLeftPoint >= 0:
                                            if i < 3:
                                                i += 1
                                            else:
                                                noSolver = True
                                                break
                                        else:
                                            up = True
                                            left = False
                                            Dx = [x0, np.max(contourObjectFiltered[:, 0])]
                                            Dy = [y0, np.max(contourObjectFiltered[:, 1])]
                                            AreaFounded = True
                                    else:
                                        if testLeftPoint >= 0:
                                            up = False
                                            left = True
                                            Dx = [np.min(contourObjectFiltered[:, 0]), x0]
                                            Dy = [np.min(contourObjectFiltered[:, 1]), y0]
                                            AreaFounded = True
                                        else:
                                            if i < 3:
                                                i += 1
                                                continue
                                            else:
                                                noSolver = True
                                                break
                            else:
                                AreaFounded = False
                                while not AreaFounded:
                                    testLeftPoint = cv.pointPolygonTest(contourObjectFiltered,
                                                                        (float(x0 - i), norm(x0 - i)), False)
                                    if cv.pointPolygonTest(contourObjectFiltered, (float(x0 + i), norm(x0 + i)),
                                                           False) >= 0:
                                        if testLeftPoint >= 0:
                                            if i < 3:
                                                i += 1
                                            else:
                                                noSolver = True
                                                break
                                        else:
                                            up = False
                                            left = False
                                            Dx = [x0, np.max(contourObjectFiltered[:, 0])]
                                            Dy = [np.min(contourObjectFiltered[:, 1]), y0]
                                            AreaFounded = True
                                    else:
                                        if testLeftPoint >= 0:
                                            up = True
                                            left = True
                                            Dx = [np.min(contourObjectFiltered[:, 0]), x0]
                                            Dy = [y0, np.max(contourObjectFiltered[:, 1])]
                                            AreaFounded = True
                                        else:
                                            if i < 3:
                                                i += 1
                                                continue
                                            else:
                                                noSolver = True
                                                break
                        IndexPointsInDx = np.logical_and(contourObjectFiltered[:, 0] > Dx[0],
                                                         contourObjectFiltered[:, 0] < Dx[1])
                        IndexPointsInDy = np.logical_and(contourObjectFiltered[:, 1] > Dy[0],
                                                         contourObjectFiltered[:, 1] < Dy[1])
                        PointsInD = contourObjectFiltered[np.logical_and(IndexPointsInDx, IndexPointsInDy)]

                        if PointsInD.shape[0] == 0:
                            continue

                        eq = norm.general_equation(PointsInD)
                        eq[np.abs(eq) < eps] = 0.
                        # Создание массива знаков
                        eq_sign = np.sign(eq)

                        if np.all(eq_sign == eq_sign[0]):
                            # Если знак не меняется, то пересечения нет
                            # raise ValueError('Line and a curve do not intersect')
                            continue

                        # Нахождение всех мест изменения знака в массиве
                        shift_eq_sign = np.roll(eq_sign, -1)
                        shift_eq_sign[-1] = eq_sign[-1]
                        eq_signchange = ((shift_eq_sign - eq_sign) != 0).astype(int)
                        change_index = np.where(eq_signchange != 0)[0]
                        if change_index.shape[0] > 1:
                            positions = [0, -1]
                        else:
                            positions = [0]
                        for position in positions:
                            if eq_sign[change_index[position]] == 0:
                                # Если первое изменение знака на ноль, то соответсвующая точка - точка пересечения
                                Points.append(PointsInD[change_index[position]])
                            else:
                                # Находим точки, в которых меняется знак, в соответствии с направлением поиска
                                [ax, ay] = PointsInD[change_index[position] + 1]
                                [bx, by] = PointsInD[change_index[position]]

                                if fabs(ax - bx) < eps:
                                    xPoint = ax
                                    yPoint = norm(xPoint)
                                elif fabs(ay - by) < eps:
                                    yPoint = ay
                                    xPoint = (yPoint - z_norm) / k_norm
                                else:
                                    # Вычисление коэффициентов для общего уравнения кривой
                                    k_segment = (by - ay) / (bx - ax)
                                    z_segment = (by - ay) * (-ax / (bx - ax) + ay / (by - ay))
                                    # Нахождение точки пересечения
                                    xPoint = (z_segment - z_norm) / (k_norm - k_segment)
                                    yPoint = (k_norm * z_segment - k_segment * z_norm) / (k_norm - k_segment)

                                Points.append(np.array([xPoint, yPoint]))

                    minPoints = None
                    minDistance = 1e+20
                    for point in Points:
                        dInPoint = self.distance(np.array([x0, y0]), point)
                        if dInPoint < minDistance:
                            minDistance = dInPoint
                            minPoints = point
                    if minDistance < 1e+20:
                        distances.append(minDistance)
                    # print(minDistance)
                except:
                    continue

        result = np.median(distances)
        print(result)
        plt.hist(distances)
        plt.show()
        return result