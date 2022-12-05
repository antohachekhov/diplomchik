import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
import FunctionsDistance

from kneed import KneeLocator
from math import ceil, log, sqrt
from scipy.signal import medfilt
from sklearn.cluster import DBSCAN
from sklearn.mixture import GaussianMixture
from sklearn.neighbors import NearestNeighbors
from statsmodels.nonparametric.api import lowess


class FractalAnalysis:
    """
    Анализатор изображений, основанный на фрактальном анализе.
    Основная цель - сегментация и измерение переходных областей на изображениях с двумя или тремя текстурами

    Атрибуты
    ----------
    _win_size : int, по умолчанию = 30
        Размер измерительного окна

    _dx : int, по умолчанию = 1
        Шаг измерительного окна по оси x

    _dy : int, по умолчанию = 1
        Шаг измерительного окна по оси y

    _show_step : bool, по умолчанию = False
        True, если необходимо отображать промежуточные результаты сегментации

    _min_samples : int, по умолчанию = 8
        Минимальное количество точек, необходимых для формирования кластера (параметр DBSCAN-алгоритма)

    _distance : float, по умолчанию = -1
        Максимальное расстояние между точками одного кластера (параметр DBSCAN-алгоритма).
        Если параметр равен -1, он будет автоматически вычислен перед DBSCAN-кластеризацией

    _min_transition : float, по умолчанию = 0.4
        Минимальная величина изменения фрактальной размерности на границе двух текстур - переходное изменение

    _median_kernel : int, по умолчанию = 21
        Ширина медианного фильтра. Должна быть нечётной!

    _method : {'cubes', 'prism'}, по умолчанию = 'prism'
        Метод измерения фрактальных размерностей

    _windows : 4-D numpy массив, по умолчанию = None
        Измерительные окна в каждой точке изображения

    _sub_windows_sizes: list, по умолчанию = None
        Размеры измерительного окна и вспомонательных окон

    _img : 2-D numpy массив, по умолчанию = None
        Изображение в пространстве оттенков серого

    field : 2-D numpy массив, по умолчанию = None
        Поле фрактальных размерностей

    distances_curves : кортеж из двух объектов - 1-D numpy массива и 3-D numpy массива, по умолчанию = None
        Кортеж из массива расстояний между кривыми и координат начала первой кривой и конца второй

    _segments : 2-D numpy массив, по умолчанию = None
        Бинарное изображение, содержащее найденные сегменты текстур

    """

    def __init__(self, show_step=False):
        """
        Инициализатор объекта класса

        """
        self._win_size = 30
        self._dx = 1
        self._dy = 1
        self._show_step = show_step
        self._min_samples = 16
        self._distance = -1
        self._min_transition = 0.5
        self._median_kernel = 31
        self._method = None
        self._windows = None
        self._sub_windows_sizes = None
        self._img = None
        self._segments = None
        self.field = None
        self.distances_curves = None

    def _check_image(self, img):
        """
        Проверка изображения

        :param img: ndarray
            Изображение
        """
        # Массив изображения должен иметь две оси (изображение в пространстве оттенков серого)
        if img.ndim != 2:
            raise ValueError('Image should be in grayscale')
        # Минимальная длина оси изображения не может быть меньше 12 (минимально допустимый размер измерительного окна)
        if min(img.shape) < 12:
            raise ValueError('Too small image')
        # Массив должен иметь значения в интервале [0, 255]
        if np.amin(img) < 0 or np.amax(img) > 255:
            raise ValueError('Invalid values in the array (values should be in range(0, 255))')

    def _check_field_parameters(self, method, win_size, dx, dy, field):
        """
        Проверка параметров поля

        :param method:  str
            Метод измерения фрактальных размерностей
        :param win_size: int
            Размер измерительного окна
        :param dx:      int
            Шаг измерительного окна по оси х
        :param dy:      int
            Шаг измерительного окна по оси y
        :param field:   2-D numpy массив
            Поле фрактальных размерностей
        """
        # значение параметра method может быть только 'cubes' или 'prism'
        if method not in ['cubes', 'prism']:
            raise ValueError("Analyses method can be 'cubes' or 'prism'")

        if field is not None:
            # размер поля должен соответсвовать размеру изображения
            if field.shape != (int((self._img.shape[0] - win_size) / dy + 1),
                               int((self._img.shape[1] - win_size) / dx + 1)):
                raise ValueError('The size of the field does not correspond to the size of the image and'
                                 ' the selected window parameters')

        try:
            # размер измерительного окна должен быть в интервале от 12 до минимального размера оси изображения
            if win_size < 12 or win_size > min(self._img.shape):
                raise ValueError('Invalid window size')
        except ValueError:
            # если введён размер меньше 12, установится размер равный 12
            if win_size < 12:
                win_size = 12
                print('win_size has been increased to the minimum allowed (12)')
            # если введён размер больше минимального размера оси изображения,
            # размер уменьшится до максимально допустимого
            elif win_size > min(self._img.shape):
                win_size = min(self._img.shape)
                print('win_size has been reduced to the maximum allowed (' + str(win_size) + ')')

        try:
            # шаги окна не могут быть меньше единицы
            if dx < 1 or dy < 1:
                raise ValueError('Invalid dx or dy')
        except ValueError:
            # если введён шаг меньше 1, установится шаг равный 1
            if dx < 1:
                dx = 1
                print('dx has been increased to the minimum allowed (1)')
            if dy < 1:
                dy = 1
                print('dy has been increased to the minimum allowed (1)')
        return win_size, dx, dy

    def _check_n_comp(self, n_comp):
        """
        Проверка числа компонент в смеси распределений, введенного пользователем

        :param n_comp : int
            Число компонент в смеси распределений

        :return: n_comp : int
            Обработанное число компонент в смеси

        """
        try:
            # число компонент может быть только 2 или 3
            if n_comp != 2 and n_comp != 3:
                raise ValueError('n_comp can be 2 or 3 only')
        except ValueError:
            # если введено число меньше 2, оно увеличивается его до 2
            if n_comp < 2:
                n_comp = 2
                print('n_comp has been changed to 2')
            # если больше 3, то оно уменьшается его до 3
            elif n_comp > 3:
                print('n_comp has been changed to 3')
                n_comp = 3
            # если в диапозоне (2, 3), оно округляется до ближайшего целого
            elif 2 < n_comp < 3:
                n_comp = round(n_comp)
                print('n_comp has been changed to ', str(n_comp))
        return n_comp

    def _check_median_kernel(self, median_kernel):
        """
        Проверка заданной пользователем ширины медианного фильтра

        :param median_kernel: int
            Ширина медианного фильтра, введённая пользователем

        :return: median_kernel: int
            Обработанная ширина медианного фильтра

        """
        # ширина медианного фильтра не может быть меньше 1
        if median_kernel < 1:
            raise ValueError('median_kernel cannot be non-positive')
        try:
            # ширина медианного фильтра должна быть нечётной
            if (median_kernel % 2) == 0:
                raise ValueError('median_kernel cannot be even')
        except ValueError:
            # если введенно чётное число, оно уменьшается или увеличивается до нечётного
            if median_kernel + 1 < min(self._img.shape):
                median_kernel += 1
            elif median_kernel - 1 > 0:
                median_kernel -= 1
            else:
                raise ValueError('Invalid value of median_kernel')
        return median_kernel

    def _check_0_1(self, value):
        """
        Проверка нахождения значения переменной в диапозоне [0, 1]

        :param value : float
            Переменная, введённая пользователем

        """
        if value < 0 or value > 1:
            raise ValueError('Value can be in range [0, 1] only')

    def _check_positive(self, value):
        """
        Проверка положительности значения переменной, введённой пользователем

        :param value : float
            Переменная, введённая пользователем

        """
        # значение не должно быть меньше или равно 0
        if value <= 0:
            raise ValueError('Value can be positive only')

    def _check_int(self, value):
        """
        Проверка значения переменной на целое число

        :param value: int
            Переменная, введённая пользователем

        """
        # для целого значения результаты приведения к типам int и float должны быть одинаковыми
        if int(value) != float(value):
            raise TypeError('Value can be integer only')

    def _generate_windows(self):
        """
        Генерация измерительных окон в каждой точке изображения

        """
        # win = np.zeros((self._win_size, self._win_size))
        # вычисляется размер массива измерительных окон
        shape = self._img.shape[:-2] + ((self._img.shape[-2] - self._win_size) // self._dy + 1,) + \
                ((self._img.shape[-1] - self._win_size) // self._dx + 1,) + (self._win_size, self._win_size)
        # вычисляется количество шагов измерительного окна
        strides = self._img.strides[:-2] + (self._img.strides[-2] * self._dy,) + \
                  (self._img.strides[-1] * self._dx,) + self._img.strides[-2:]
        self._windows = np.lib.stride_tricks.as_strided(self._img, shape=shape, strides=strides)

    def _method_cubes(self):
        """
        Измерение фрактальных размерностей с помощью метода кубов

        Метод основан на подсчёте количества кубов, необходимых для полного покрытия поверхности на разных масштабах

        """
        self.field = np.zeros((self._windows.shape[0], self._windows.shape[1]))

        # проход по каждому измерительному окну
        for i in range(self._windows.shape[0]):
            for j in range(self._windows.shape[1]):
                t_win = self._windows[i][j]

                # массив количества кубов для разных размеров субокон
                n = np.zeros(self._sub_windows_sizes.shape)

                # проход по каждому размеру вспомогательных окон
                for ei in range(self._sub_windows_sizes.shape[0]):
                    size = self._sub_windows_sizes[ei]

                    # переменная для подсчёта количества кубов в пределах измерительного окна
                    n_e = 0

                    # количество субокон по оси х и у в измерительном окне
                    n_x = ceil(t_win.shape[1] / size)
                    n_y = ceil(t_win.shape[0] / size)

                    # проход по каждому субокну
                    for i_y in range(n_y):
                        for i_x in range(n_x):

                            # вычисление координат субокна
                            y_beg = i_y * size
                            y_end = (i_y + 1) * size
                            if y_end > t_win.shape[0]:
                                y_end = t_win.shape[0]
                            x_beg = i_x * size
                            x_end = (i_x + 1) * size
                            if x_end > t_win.shape[1]:
                                x_end = t_win.shape[1]

                            # выделение субокна
                            cut = t_win[y_beg:y_end, x_beg:x_end]

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
                lgn = np.zeros(n.shape)
                lge = np.zeros(self._sub_windows_sizes.shape)
                for k in range(self._sub_windows_sizes.shape[0]):
                    lgn[k] = log(n[k])
                    lge[k] = log(self._sub_windows_sizes[k])

                # по МНК находим наклон регрессии логарифма количества кубов от логарифма размера субокон
                A = np.vstack([lge, np.ones(len(lge))]).T
                D, _ = np.linalg.lstsq(A, lgn, rcond=None)[0]

                # отрицательная величина наклона регрессии - величина фрактальной размерности в данной точке
                self.field[i][j] = -D

            print('\rMethod of cubes [', '█' * int(i / self._windows.shape[0] * 20),
                  ' ' * int((self._windows.shape[0] - i) / self._windows.shape[0] * 20),
                  ']\t', i, '\t/ ',
                  self._windows.shape[0] - 1,
                  end='')
        print('\n')

    def _method_prism(self):
        """
        Измерение фрактальных размерностей с помощью метода призм

        Метод основан на измерении площади поверхности призм, наложенных на поверхность при разных масштабах

        """
        self.field = np.zeros((self._windows.shape[0], self._windows.shape[1]))

        # проход по каждому измерительному окну
        for i in range(self._windows.shape[0]):
            for j in range(self._windows.shape[1]):
                t_win = self._windows[i][j]

                # массив значений площади поверхности призм при разных размерах субокон
                Se = np.zeros(self._sub_windows_sizes.shape)

                # проход по каждому размеру субокна
                for si in range(self._sub_windows_sizes.shape[0]):
                    size = self._sub_windows_sizes[si]

                    # переменная для подсчёта площади поверхностей призм в окне при текущих субокнах
                    S = 0

                    # количество субокон по оси х и у
                    n_x = ceil(t_win.shape[1] / size)
                    n_y = ceil(t_win.shape[0] / size)

                    # проход по каждому субокну
                    for i_y in range(n_y):
                        for i_x in range(n_x):
                            y_beg = i_y * size
                            y_end = (i_y + 1) * size
                            if y_end > t_win.shape[0]:
                                y_end = t_win.shape[0]
                            x_beg = i_x * size
                            x_end = (i_x + 1) * size
                            if x_end > t_win.shape[1]:
                                x_end = t_win.shape[1]
                            cut = t_win[y_beg:y_end, x_beg:x_end]

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

                # логарифмирование площади поверхностей призм и размеры субокон
                lgS = np.zeros(Se.shape)
                lge = np.zeros(self._sub_windows_sizes.shape)
                for k in range(self._sub_windows_sizes.shape[0]):
                    lgS[k] = log(Se[k])
                    lge[k] = log(self._sub_windows_sizes[k])

                # по МНК находится наклон регрессии логарифма площади от логарифма размера субокна
                A = np.vstack([lge, np.ones(len(lge))]).T
                D, _ = np.linalg.lstsq(A, lgS, rcond=None)[0]

                # отрицательный наклон регрессии + 2 - есть фрактальная размерность в данной точке
                self.field[i][j] = 2 - D

            print('\rMethod of prism [', '█' * int(i / self._windows.shape[0] * 20),
                  ' ' * int((self._windows.shape[0] - i) / self._windows.shape[0] * 20), ']\t', i, '\t/ ',
                  self._windows.shape[0] - 1,
                  end='')
        print('\n')

    def _generate_field(self):
        """
        Функция генерации поля фрактальных размерностей

        """
        self._generate_windows()

        self._sub_windows_sizes = np.array([self._win_size,
                                            int((self._win_size + 1) / 2),
                                            int((self._win_size + 3) / 4)])

        if self._method == 'cubes':
            self._method_cubes()
        else:
            self._method_prism()

    def set_field(self,
                  img,
                  win_size,
                  dx,
                  dy,
                  method='prism',
                  field=None):
        """
        Функция инициализации поля фрактальных размерностей.
        Если поле не задано пользователем, оно генерируется с заданными параметрами.
        В ином случае, загружается заданное пользователем поле.

        :param img: 2-D numpy массив
            Изображение

        :param win_size: int
            Размер измерительного окна

        :param dx: int
            Шаг измерительного окна по оси х

        :param dy: int
            Шаг измерительного окна по оси у

        :param method: str, по умолчанию = 'prism'
            Метод измерения фрактальной размерности

        :param field: 2-D numpy массив, по умолчанию = None
            Поле фрактальных размерностей для изображения img.
            Если не задано, оно будет вычислено в соответствии с заданными параметрами.

        """
        self._check_image(img)
        # если изображение успешно проверено, оно загружается в объект класса
        self._img = img
        self._segments = np.zeros(self._img.shape, dtype=np.uint8)

        # проверка параметров поля и загрузка их в объект класса
        self._win_size, self._dx, self._dy = self._check_field_parameters(method, win_size, dx, dy, field)
        self.field = field
        self._method = method

        # генерация поля, если оно не задано
        if self.field is None:
            self._generate_field()

    def _find_extr(self, data):
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

    def _segment_field_change(self):
        """
        Сегментация переходного слоя по сильным изменения поля

        """
        # массивы точек начала и конца сильных изменений соответственно
        change_begin = np.array([0, 0], dtype=np.uint32)
        change_end = np.array([0, 0], dtype=np.uint32)

        '''
        Для исследований!

        print('Show filter? 1 - Yes, 0 - No')
        sf = int(input())
        sfcol = -1
        if sf:
            print('Choose column or write -1  ', end='')
            plt.imshow(cv.cvtColor(self._img[int(self._win_size / 2): -int(self._win_size / 2) + 1,
                                   int(self._win_size / 2): -int(self._win_size / 2) + 1], cv.COLOR_GRAY2RGB))
            plt.xticks(np.arange(0, self._img.shape[1] - self._win_size, 50))
            plt.show()
            sfcol = int(input())
            
        '''

        # проход по каждому столбцу в поле
        for n, column in enumerate(self.field.T):  # range(self.field.shape[1]):
            # column = self.field[:, col]

            # медианная оценка выборки
            filter_column = medfilt(column, kernel_size=self._median_kernel)

            # координаты локальных экстремумов и разность значений между ними
            amplMF, extrMF = self._find_extr(filter_column)

            """
            if sf and n == sfcol:
                ampl, extr = self._find_extr(column)

                fig, axis = plt.subplots(1, 2)
                axis[0].bar(extr[1:], ampl, color='purple')
                axis[0].plot(column, color='#FF7F00', linewidth=1)

                axis[1].bar(extrMF[1:], amplMF, color='blue')
                axis[1].plot(filter_column, color='#FF7F00', linewidth=1)

                plt.show()
                plt.close()
            """

            # Выделение областей, в которых разность между экстремумами больше,
            # чем величина изменения фрактальной размерности на границе двух текстур
            ind_pick = np.where(amplMF > self._min_transition)
            if ind_pick[0].size > 0:
                for j in ind_pick[0]:
                    # области на границе изображения не берутся во внимание,
                    # так как на границах поле искажено
                    if j != 0 and j < extrMF.shape[0] - 1:
                        if 0 < extrMF[j] < self.field.shape[0] - 1 and extrMF[j + 1] < self.field.shape[0] - 1:
                            change_begin = np.vstack([change_begin, np.array([n, extrMF[j]])])
                            change_end = np.vstack([change_end, np.array([n, extrMF[j + 1]])])

        change_begin = np.delete(change_begin, 0, axis=0)
        change_end = np.delete(change_end, 0, axis=0)

        '''
        Для исследования!
        
        imgRGB = cv.cvtColor(self._img, cv.COLOR_GRAY2RGB)
        plt.imshow(imgRGB)
        plt.scatter(change_begin[:, 0], change_begin[:, 1], c='red', s=1)
        plt.scatter(change_end[:, 0], change_end[:, 1], c='red', s=1)
        plt.show()
        '''

        # перевод точек переходного изменения поля в размерность изображения
        for point in change_begin:
            point[0] = int(point[0] * self._dx + self._win_size / 2)
            point[1] = int(point[1] * self._dy + self._win_size / 2)

        for point in change_end:
            point[0] = int(point[0] * self._dx + self._win_size / 2)
            point[1] = int(point[1] * self._dy + self._win_size / 2)

        # Оценка параметра eps алгоритма DBSCAN
        nearest_neighbors = NearestNeighbors(n_neighbors=11)
        neighbors = nearest_neighbors.fit(change_begin)
        distances, indices = neighbors.kneighbors(change_begin)
        distances = np.sort(distances[:, 10], axis=0)
        i = np.arange(len(distances))
        knee = KneeLocator(i, distances, S=1, curve='convex', direction='increasing', interp_method='polynomial')

        # если пользователем не задано другое значение distance
        if self._distance == -1.:
            self._distance = 1. * distances[knee.knee]

        # кластеризация точек, соответствующих началу изменения
        dbscan_cluster = DBSCAN(eps=self._distance, min_samples=self._min_samples)
        dbscan_cluster.fit(change_begin)

        # получение списка номеров кластеров и их размеров
        uniq_labels, label_count = np.unique(dbscan_cluster.labels_, return_counts=True)

        # средний размер кластеров, не считая кластер шума
        label_count_mean = np.mean(label_count[1:])

        # массивы для точек из не отброшенных кластеров
        clustered_change_begin = np.array([0, 0], dtype=np.uint32)
        clustered_change_end = np.array([0, 0], dtype=np.uint32)
        filtered_labels = np.empty(0, dtype=np.uint8)

        # сохранение точек, размер кластера которых больше среднего размера всех кластеров
        for i in range(1, uniq_labels.shape[0]):
            if label_count[i] >= label_count_mean:
                clustered_change_begin = np.vstack([clustered_change_begin,
                                                    change_begin[dbscan_cluster.labels_ == uniq_labels[i]]])
                clustered_change_end = np.vstack([clustered_change_end,
                                                  change_end[dbscan_cluster.labels_ == uniq_labels[i]]])
                filtered_labels = np.append(filtered_labels, [i] * label_count[i])

        clustered_change_begin = np.delete(clustered_change_begin, 0, axis=0)
        clustered_change_end = np.delete(clustered_change_end, 0, axis=0)

        # перевод изображения в цветовое пространство RGB
        imgRGB = cv.cvtColor(self._img, cv.COLOR_GRAY2RGB)

        if self._show_step:
            fig, axis = plt.subplots(2, 2)

            axis[0, 0].imshow(imgRGB)
            axis[0, 0].scatter(change_begin[:, 0], change_begin[:, 1], c='red', s=1)
            axis[0, 0].scatter(change_end[:, 0], change_end[:, 1], c='red', s=1)
            axis[0, 0].set_title("Image and amplitude's points")

            axis[0, 1].set_title('Knee Point = ' + str(self._distance))
            axis[0, 1].plot(knee.x, knee.y, "b", label="data")
            axis[0, 1].vlines(knee.knee, plt.ylim()[0], plt.ylim()[1], linestyles="--", label="knee/elbow")
            axis[0, 1].legend(loc="best")

            axis[1, 0].scatter(change_begin[:, 0], change_begin[:, 1], c=dbscan_cluster.labels_, s=15)
            axis[1, 0].scatter(change_end[:, 0], change_end[:, 1], c=dbscan_cluster.labels_, s=15)
            axis[1, 0].set_title('DBSCAN eps=' + str(self._distance) + ' min_samples=' + str(self._min_samples))

            axis[1, 1].scatter(clustered_change_begin[:, 0], clustered_change_begin[:, 1], c=filtered_labels, s=10)
            axis[1, 1].scatter(clustered_change_end[:, 0], clustered_change_end[:, 1], c=filtered_labels, s=10)
            axis[1, 1].set_title('[Filtered] DBSCAN eps=' + str(self._distance) + ' m_samp=' + str(self._min_samples))

            plt.show()
            plt.close()

        n_cluster = np.unique(filtered_labels).shape[0]
        distances_of_curves = (np.zeros(shape=n_cluster), np.zeros(shape=(n_cluster, 2, 2)))
        i_cluster = 0

        # проход по каждому кластеру
        for i in np.unique(filtered_labels):
            x = clustered_change_begin[filtered_labels == i, 0]
            y1 = clustered_change_begin[filtered_labels == i, 1]
            y2 = clustered_change_end[filtered_labels == i, 1]

            # строится регрессионная модель для точек, соответствующих началу переходного изменения поля
            z1 = lowess(y1, x, frac=0.1)
            # строится регрессионная модель для точек, соответствующих концу переходного изменения поля
            z2 = lowess(y2, x, frac=0.1)

            distances_of_curves[0][i_cluster] = FunctionsDistance.functions_distance(z1, z2)

            plt.plot(z1[:, 0], z1[:, 1], '--', c='blue', linewidth=3)
            plt.plot(z2[:, 0], z2[:, 1], '--', c='blue', linewidth=3)
            plt.annotate('{}'.format(distances_of_curves[0][i_cluster]), xy=(z1[0, 0], z1[0, 1]),
                         xytext=(z1[0, 0], z1[0, 1]), arrowprops={'facecolor': 'white', 'shrink': 0.1})

            # округляем точки до целых, чтобы поставить им в соответствие пиксели изображения
            points_beg = np.around(z1).astype(np.int32)
            points_beg = points_beg.reshape((-1, 1, 2))
            points_end = np.around(z2).astype(np.int32)
            points_end = np.flipud(points_end).reshape((-1, 1, 2))
            pts = np.concatenate([points_beg, points_end, points_beg[0].reshape((-1, 1, 2))])

            distances_of_curves[1][i_cluster, 0] = points_beg[0][0]
            distances_of_curves[1][i_cluster, 1] = points_end[0][0]

            # рисуем на маске линии начала и конца изменения и закрашиваем область между ними
            cv.fillPoly(self._segments, [pts], 255)

            i_cluster += 1

        plt.show()

        self.distances_curves = distances_of_curves

    def _segment_distribution(self, img, intensity):
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
        value_img = cv.inRange(img, intensity, intensity + 1)

        # "открытие" изображения (удаление мелких объектов) с эллиптическим ядром 7х7
        kernel = cv.getStructuringElement(cv.MORPH_RECT, (7, 7))
        morph_open = cv.morphologyEx(value_img, cv.MORPH_OPEN, kernel)

        # "закрытие" изображения (заполнение мелких пропусков в объектах) с эллиптическим ядром 5х5
        kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE, (5, 5))
        morph_close = cv.morphologyEx(morph_open, cv.MORPH_CLOSE, kernel)

        # выделение объектов на изображении и составление статистики
        contours, hierarchy = cv.findContours(morph_close.copy(), cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
        cv.drawContours(morph_close, contours, -1, 100, 2, cv.LINE_AA, hierarchy, 3)
        nb_components, output, stats, centroids = cv.connectedComponentsWithStats(morph_close, 4, cv.CV_32S)

        # вычисление средней площади всех объектов
        meanS = np.mean(stats[1:, -1])

        # массив для маски с областями для сегментации
        img2 = np.zeros(output.shape, dtype=np.uint8)

        # заполнение маски
        for i in range(1, nb_components):
            # заносятся только те объекты, площадь которых больше средней площади всех объектов
            if stats[i, -1] > meanS:
                img2[output == i] = 255

        # наращивание объектов
        radius = int(self._win_size / 2 if self._win_size / 2 % 2 == 1 else self._win_size / 2 - 1)
        kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE, (radius, radius))
        result = cv.dilate(img2, kernel)

        if self._show_step:
            plt.subplot(2, 3, 1)
            plt.imshow(img, cmap='gray')
            plt.title('Original')
            plt.xticks([])
            plt.yticks([])

            plt.subplot(2, 3, 2)
            plt.imshow(value_img, cmap='gray')
            plt.title('Range')
            plt.xticks([])
            plt.yticks([])

            plt.subplot(2, 3, 3)
            plt.imshow(morph_open, cmap='gray')
            plt.title('Open')
            plt.xticks([])
            plt.yticks([])

            plt.subplot(2, 3, 4)
            plt.imshow(morph_close, cmap='gray')
            plt.title('Close')
            plt.xticks([])
            plt.yticks([])

            plt.subplot(2, 3, 5)
            plt.imshow(img2, cmap='gray')
            plt.title('Sort')
            plt.xticks([])
            plt.yticks([])

            plt.subplot(2, 3, 6)
            plt.imshow(result, cmap='gray')
            plt.title('Blur&Thresh')
            plt.xticks([])
            plt.yticks([])

            plt.show()

        return result

    def _segment_min_distribution(self, n_comp):
        """
        Сегментация переходного слоя через EM-классификацию

        :param n_comp: int
            Количество компонент в смеси распределений

        """
        # классификация значений поля по EM-алгоритму с тремя компонентами в смеси
        X = np.expand_dims(self.field.flatten(), 1)
        model = GaussianMixture(n_components=n_comp, covariance_type='full')
        model.fit(X)
        predict = model.predict(X)
        means = model.means_
        field_predict = np.resize(predict, self.field.shape)

        # перевод поля значений классов в пространство оттенков серого
        field_pr_img = field_predict / (n_comp - 1) * 255
        field_pr_img = field_pr_img.astype(np.uint8)

        # cv.imshow('EM', field_pr_img)

        # определение какому классу соответствуют значения, принадлежащие классу с минимальным математическим ожиданием
        stratum_class = np.argmin(means)
        intensive_stratum = int(stratum_class / (n_comp - 1) * 255)

        # выделение областей, принадлежащих классу с минимальным математическим ожиданием
        # получаем маску
        mask_stratum = self._segment_distribution(field_pr_img, intensive_stratum)

        # переносим получившуюся маску поля на маску для изображения
        mask_stratum = cv.resize(mask_stratum,
                                 np.flip((self._img.shape[0] - self._win_size + 1,
                                          self._img.shape[1] - self._win_size + 1)),
                                 fx=self._dx, fy=self._dy, interpolation=cv.INTER_LINEAR_EXACT)
        self._segments[int(self._win_size / 2): -int(self._win_size / 2) + 1,
        int(self._win_size / 2): -int(self._win_size / 2) + 1] = mask_stratum

    def segment_stratum(self,
                        n_comp,
                        median_kernel=None,
                        min_transition=None,
                        distance=None,
                        min_samples=None
                        ):
        """
        Функция для выделения переходного слоя на изображении

        :param n_comp: int
            Количество компонент в смеси распределений фрактальных размерностей.

        :param median_kernel: int, по умолчанию = None
            Ширина медианного фильтра, введённая пользователем

        :param min_transition: float, по умолчанию = None
            Минимальная величина переходного изменения

        :param distance: float, по умолчанию = None
            Максимальное расстояние между точками одного кластера, введённое пользователем

        :param min_samples: int, по умолчанию = None
            Минимальное количество точек, необходимых для формирования кластера, введённое пользователем

        """
        n_comp = self._check_n_comp(n_comp)

        if median_kernel is not None:
            self._median_kernel = self._check_median_kernel(median_kernel)
        if min_transition is not None:
            self._check_0_1(min_transition)
            self._min_transition = min_transition
        if distance is not None:
            self._check_positive(distance)
            self._distance = distance
        if min_samples is not None:
            self._check_positive(min_samples)
            self._check_int(min_samples)
            self._min_samples = min_samples

        if n_comp == 2:
            # если количество компонент в смеси распределений - 2,
            # то сегментация проходит по сильным изменениям размерности
            self._segment_field_change()
        else:
            # если количество компонент в смеси распределений - 3,
            # то сегментация проходит по выделению областей,
            # принадлежащим распределению с минимальным математическим ожиданием
            self._segment_min_distribution(n_comp)

        return self._segments
