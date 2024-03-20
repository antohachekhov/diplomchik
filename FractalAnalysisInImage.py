from math import sqrt, ceil, log
import numpy as np

def trianglePrism(window, subWindowsSizes):
    """
    Измерение фрактальных размерностей с помощью метода треугольных призм

    Метод основан на измерении площади поверхности призм, наложенных на поверхность при разных масштабах

    """
    # массив значений площади поверхности призм при разных размерах субокон
    Se = np.zeros(subWindowsSizes.shape)

    # проход по каждому размеру субокна
    for si, size in enumerate(subWindowsSizes):
        # переменная для подсчёта площади поверхностей призм в окне при текущих субокнах
        S = 0

        # количество субокон по оси х и у
        n_x = ceil(window.shape[1] / size)
        n_y = ceil(window.shape[0] / size)

        # проход по каждому субокну
        for i_y in range(n_y):
            for i_x in range(n_x):
                y_beg = i_y * size
                y_end = (i_y + 1) * size
                if y_end > window.shape[0]:
                    y_end = window.shape[0]
                x_beg = i_x * size
                x_end = (i_x + 1) * size
                if x_end > window.shape[1]:
                    x_end = window.shape[1]
                cut = window[y_beg:y_end, x_beg:x_end]

                # ширина, высота и диагональ субокна
                sx = cut.shape[1]
                sy = cut.shape[0]
                sdi = sqrt(sx ** 2 + sy ** 2)

                # каждая точка призмы находится на высоте интенсивности яркости в этой точке
                a = int(cut[0][0])
                b = int(cut[0][sx - 1])
                c = int(cut[sy - 1][0])
                d = int(cut[sy - 1][sx - 1])
                # e = int(cut[int((sy - 1) / 2)][int((sx - 1) / 2)])
                e = (a + b + c + d) / 4

                # вычисление рёбра от каждой точки до центра субокна
                ae = sqrt((a - e) ** 2 + (sdi / 2) ** 2)
                be = sqrt((b - e) ** 2 + (sdi / 2) ** 2)
                ce = sqrt((c - e) ** 2 + (sdi / 2) ** 2)
                de = sqrt((d - e) ** 2 + (sdi / 2) ** 2)

                # вычисление рёбер между точками
                ab = sqrt((b - a) ** 2 + sx ** 2)
                bd = sqrt((b - d) ** 2 + sy ** 2)
                cd = sqrt((c - d) ** 2 + sx ** 2)
                ac = sqrt((a - c) ** 2 + sy ** 2)

                # вычисление периметра каждого треугольника
                pA = (ab + be + ae) / 2
                pB = (bd + be + de) / 2
                pC = (cd + de + ce) / 2
                pD = (ac + ce + ae) / 2

                # вычисление площади каждого треугольника
                SA = sqrt(pA * (pA - ab) * (pA - be) * (pA - ae))
                SB = sqrt(pB * (pB - bd) * (pB - be) * (pB - de))
                SC = sqrt(pC * (pC - cd) * (pC - de) * (pC - ce))
                SD = sqrt(pD * (pD - ac) * (pD - ce) * (pD - ae))

                # вычисление площади поверхности всей призмы с остальными в измерительном окне
                S += SA + SB + SC + SD
        Se[si] = S

    # логарифмирование площадей поверхностей призм и размеров субокон
    lgS = np.log(Se)
    lge = np.log(subWindowsSizes)

    # по МНК находится наклон регрессии логарифма площади от логарифма размера субокна
    A = np.vstack([lge, np.ones(len(lge))]).T
    D, _ = np.linalg.lstsq(A, lgS, rcond=None)[0]

    # отрицательный наклон регрессии + 2 - есть фрактальная размерность в данной точке
    return 2 - D


def cubes(window, subWindowsSizes):
    """
    Измерение фрактальных размерностей с помощью метода кубов

    Метод основан на подсчёте количества кубов, необходимых для полного покрытия поверхности на разных масштабах

    """
    # массив количества кубов для разных размеров субокон
    n = np.zeros(subWindowsSizes.shape)

    # проход по каждому размеру вспомогательных окон
    for ei, size in enumerate(subWindowsSizes):
        size = subWindowsSizes[ei]

        # переменная для подсчёта количества кубов в пределах измерительного окна
        n_e = 0

        # количество субокон по оси х и у в измерительном окне
        n_x = ceil(window.shape[1] / size)
        n_y = ceil(window.shape[0] / size)

        # проход по каждому субокну
        for i_y in range(n_y):
            for i_x in range(n_x):

                # вычисление координат субокна
                y_beg = i_y * size
                y_end = (i_y + 1) * size
                if y_end > window.shape[0]:
                    y_end = window.shape[0]
                x_beg = i_x * size
                x_end = (i_x + 1) * size
                if x_end > window.shape[1]:
                    x_end = window.shape[1]

                # выделение субокна
                cut = window[y_beg:y_end, x_beg:x_end]

                # по максимальному и минимальному значению интенсивности яркости пикселей
                # внутри субокна вычисляется количество кубов размера субокна, необходимых
                # для покрытия поверхности внутри этого субокна
                max_in_cut = np.amax(cut)
                min_in_cut = np.amin(cut)
                if max_in_cut != min_in_cut:
                    n_e += ceil(max_in_cut / size) - ceil(min_in_cut / size) + 1
                else:
                    n_e += 1
        n[ei] = n_e

    # логарифмируем количество размеров субокон и количество кубов для разных размеров субокон
    lgn = np.log(n)
    lge = np.log(subWindowsSizes)

    # по МНК находим наклон регрессии логарифма количества кубов от логарифма размера субокон
    A = np.vstack([lge, np.ones(len(lge))]).T
    D, _ = np.linalg.lstsq(A, lgn, rcond=None)[0]

    # отрицательная величина наклона регрессии - величина фрактальной размерности в данной точке
    return -D
