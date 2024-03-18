import numpy as np
from math import dist, fabs


class Line:
    """
        Объект класса - прямая линия,
        описываемая каноническим уравнением прямой:
        ax + by + c = 0

        Атрибуты
        ----------
        a : float, по умолчанию = 1
            Коэффициент при x в каноническом уравнении прямой

        b : float, по умолчанию = 1
            Коэффициент при y в каноническом уравнении прямой

        с : float, по умолчанию = 0
            Свободный коэффициент в каноническом уравнении прямой
    """

    def __init__(self, a=1., b=1., c=0.):
        self.a = a
        self.b = b
        self.c = c

    def __call__(self, x, *args, **kwargs):
        """
        Возвращает значение y прямой при заданной координате x

        :param x: float
            Значение x, в котором нужно найти точку прямой
        """
        return (self.a / self.b) * x + (self.c / self.b)

    def y(self, x):
        """
        Возвращает значение y прямой при заданной координате x

        :param x: float
            Значение x, в котором нужно найти точку прямой
        """
        return (self.a / self.b) * x + (self.c / self.b)

    def general_equation(self, points):
        """
        Возвращает значение канонического уравнения прямой для заданной точки (x, y)

        :param points: ndarray
            Точка или набор точек (x, y), для которых нужно найти значение канонического уравнения
        """
        if points.ndim == 1:
            # points - одна точка
            if points.shape[0] != 2:
                # points не является точкой в двумерном пространстве
                raise ValueError('Number of features must be 2')
            else:
                return self.a * points[0] - self.b * points[1] + self.c
        elif points.ndim == 2:
            # points - набор точек
            if points.shape[1] != 2:
                # points не являются точками в двумерном пространстве
                raise ValueError('Number of features must be 2')
            else:
                return self.a * points[:, 0] - self.b * points[:, 1] + self.c
        elif points.ndim == 3:
            # points - набор массивов с одной точкой
            if points.shape[2] != 2:
                # points не являются точками в двумерном пространстве
                raise ValueError('Number of features must be 2')
            else:
                return self.a * points[:, :, 0] - self.b * points[:, :, 1] + self.c
        else:
            raise ValueError('Points have incorrect dimension')


def intersection_line_curve(k_line, z_line, curve, eps=1e-7):
    """
    Поиск точки пересечения прямой и кривой

    :param k_line: float
        Коэффициент k при x в общем уравнении прямой
    :param z_line: float
        Свободный коэффициент в общем уравнении прямой
    :param curve: ndarray
        Набор точек кривой, заданной на дискретной сетке
    :param eps: float, по умолчанию = 1е-7
        Точность вычислений
    """
    if curve.shape[1] != 2:
        # Кривая задана не в двумерном пространстве
        raise ValueError('Number of features must be 2')

    if k_line > 0:
        # Прямая наклонена вправо
        # sign_direction = 0, если необходимо найти первое изменение знака
        sign_direction = 0
        curve_direction = -1

    else:
        # Прямая наклонена влево
        # sign_direction = -1, если необходимо найти последнее изменение знака
        sign_direction = -1
        curve_direction = 1
    # curve_direction - дополнительная переменная для нахождения точек отрезка AB

    # Создание объекта класса линий для прямой
    lineModel = Line(a=k_line, c=z_line)
    # Находим значение канонического уравнения для всех точек кривой
    eq = lineModel.general_equation(curve)
    eq[np.abs(eq) < eps] = 0.
    # Создание массива знаков
    eq_sign = np.sign(eq)

    if np.all(eq_sign == eq_sign[0]):
        # Если знак не меняется, то пересечения нет
        raise ValueError('Line and a curve do not intersect')

    # Нахождение всех мест изменения знака в массиве
    eq_signchange = ((np.roll(eq_sign, curve_direction) - eq_sign) != 0).astype(int)
    change_index = np.where(eq_signchange == 1)[0]

    if eq_sign[change_index[sign_direction]] == 0:
        # Если первое изменение знака на ноль, то соответсвующая точка - точка пересечения
        p = curve[change_index[sign_direction]]
    else:
        # Находим точки, в которых меняется знак, в соответствии с направлением поиска
        a = curve[change_index[sign_direction] - curve_direction]
        b = curve[change_index[sign_direction]]

        if (b[0] - a[0]) == 0. or (b[1] - a[1]) == 0.:
            # Если точки в одной плоскости X или Y, то решения нет
            raise ValueError('Error')
        else:
            # Вычисление коэффициентов для общего уравнения кривой
            k_segment = (b[1] - a[1]) / (b[0] - a[0])
            z_segment = (b[1] - a[1]) * (-a[0] / (b[0] - a[0]) + a[1] / (b[1] - a[1]))

        # Нахождение точки пересечения
        p = np.array([(z_segment - z_line) / (k_line - k_segment),
                      (k_line * z_segment - k_segment * z_line) / (k_line - k_segment)])

    return p


def functions_distance_normal(curve1, curve2, eps=1e-7, n_diff=3):
    """
    Вычисление расстояния между кривыми с помощью построения нормалей

    :param curve1: ndarray
        Набор точек, задающие кривую-1
    :param curve2: ndarray
        Набор точек, задающие кривую-2
    :param eps: float, по умолчанию = 1е-7
        Точность вычислений
    :param n_diff: int, по умолчанию = 3
        Шаг дифференцирования
    """
    distanceList = np.empty(shape=0)

    if curve1.shape[0] < 2 or curve2.shape[0] < 2:
        # Одна из кривых задана одной единственной точкой
        raise ValueError('Number of function points must be at least 2')
    if curve1.shape[1] != 2 or curve2.shape[1] != 2:
        # Одна из кривых задана не в двумерном пространстве
        raise ValueError('Number of features must be 2')

    # Средний шаг по сетке для кривой-1
    hx = np.mean(np.diff(curve1[:, 0]))
    # Переменная для вычисления среднего расстояния между кривыми
    # distance = 0.
    # Количество найденных точек пересечения
    # n = 0
    # Вычисление расстояния для каждой точки кривой-1
    for (index,), x0 in np.ndenumerate(curve1[:, 0]):
        if index < n_diff:
            # Ищем правую производную
            [dx, dy] = curve1[index + n_diff] - curve1[index]
        elif index >= curve1.shape[0] - n_diff:
            # Ищем левую производную
            [dx, dy] = curve1[index] - curve1[index - n_diff]
        else:
            # Ищем центральную производную
            [dx, dy] = curve1[index + n_diff] - curve1[index - n_diff]

        if fabs(dy) < eps:
            # Если производная равно 0, то нормаль направлена строго вверх
            # print("In x = {} the normal is directed strictly vertically".format(x0))
            # Нахождение точки в кривой-2 максимально близкой к координате x при которой построена нормаль
            difarray = np.abs(curve2[:, 0] - x0)
            index = np.argmin(difarray)
            if difarray[index] <= hx:
                d = dist(curve1[index], curve2[index])
                distanceList = np.append(distanceList, [d], axis=0)
                # n += 1
                continue
        if fabs(dx) < eps:
            # Если производная стремится к бесконечности, то решения нет
            print("In x = {} the normal is directed strictly horizontally".format(x0))
            continue

        # Вычисление коэффициентов k и z из общего уравнения прямой
        k_norm = -1. / (dy / dx)
        z_norm = curve1[index, 1] + x0 / (dy / dx)

        # Задаём область определения нормали
        if k_norm > 0:
            # Нормаль направлена вправо
            x_min = x0
            x_max = curve2[-1, 0]

        else:  # k_line < 0
            # Нормаль направлена влево
            x_min = curve2[0, 0]
            x_max = x0

        # Определение множества точек кривой-2, заданных в области определения нормали
        curve2range = curve2[np.logical_and(x_min <= curve2[:, 0], x_max >= curve2[:, 0])]

        # Поиск точки пересечения нормали и кривой-2
        try:
            p = intersection_line_curve(k_norm, z_norm, curve2range)
        except Exception:
            continue
        # Находим расстояние между точками пересечения и основания нормали
        d = dist(curve1[index], p)
        # distance += d # НАДО БУДЕТ СДЕЛАТЬ ЧТОБЫ ФУНКЦИЯ ВОЗВРАЩАЛА НЕ ОБЩЕЕ РАССТОЯНИЕ А В КАЖДОЙ ТОЧКЕ
        distanceList = np.append(distanceList, [d], axis=0)
        # n += 1

    # Находим среднее расстояние
    # distance /= n
    # return distance, distanceList
    return distanceList


def functions_distance_vertical(curve1, curve2):
    """
    Вычисление расстояния между кривыми для точек в одной плоскости Х

    :param curve1: ndarray
        Набор точек, задающие кривую-1
    :param curve2: ndarray
        Набор точек, задающие кривую-2
    """
    if curve1.shape[0] < 2 or curve2.shape[0] < 2:
        # Одна из кривых задана одной единственной точкой
        raise ValueError('Number of function points must be at least 2')
    if curve1.shape[1] != 2 or curve2.shape[1] != 2:
        # Одна из кривых задана не в двумерном пространстве
        raise ValueError('Number of features must be 2')

    # Находим среднее расстояние между точками кривых
    mean = np.mean(np.abs(curve2[:, 1] - curve1[:, 1]))
    return mean
