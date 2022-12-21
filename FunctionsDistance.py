import numpy as np
from math import dist, fabs

class Line:

    def __init__(self, a=1., b=1., c=0.):
        self.a = a
        self.b = b
        self.c = c

    def y(self, x):
        return (self.a / self.b) * x + (self.c / self.b)

    def general_equation(self, points):
        if points.ndim == 1:
            if points.shape[0] != 2:
                raise ValueError('Number of features must be 2')
            else:
                return self.a * points[0] - self.b * points[1] + self.c
        elif points.ndim == 2:
            if points.shape[1] != 2:
                raise ValueError('Number of features must be 2')
            else:
                return self.a * points[:, 0] - self.b * points[:, 1] + self.c
        else:
            raise ValueError('Points have incorrect dimension')


def intersection_line_curve(k_line, z_line, curve, eps=1e-7):
    if curve.shape[1] != 2:
        raise ValueError('Number of features must be 2')

    if k_line > 0:
        # Right
        sign_direction = 0
        curve_direction = -1

    else:  # k_line < 0
        # Left
        sign_direction = -1
        curve_direction = 1

    lineModel = Line(a=k_line, c=z_line)
    eq = lineModel.general_equation(curve)
    eq[np.abs(eq) < eps] = 0.
    eq_sign = np.sign(eq)

    if np.all(eq_sign == eq_sign[0]):
        raise ValueError('Line and a curve do not intersect')

    eq_signchange = ((np.roll(eq_sign, curve_direction) - eq_sign) != 0).astype(int)
    change_index = np.where(eq_signchange == 1)[0]

    if eq_sign[change_index[sign_direction]] == 0:
        p = curve[change_index[sign_direction]]
    else:
        a = curve[change_index[sign_direction] - curve_direction]
        b = curve[change_index[sign_direction]]

        if (b[0] - a[0]) == 0. or (b[1] - a[1]) == 0.:
            raise ValueError('Error')
        else:
            k_segment = (b[1] - a[1]) / (b[0] - a[0])
            z_segment = (b[1] - a[1]) * (-a[0] / (b[0] - a[0]) + a[1] / (b[1] - a[1]))

        p = np.array([(z_segment - z_line) / (k_line - k_segment),
                      (k_line * z_segment - k_segment * z_line) / (k_line - k_segment)])

    return p



def functions_distance(curve1, curve2, eps=1e-7, n_diff=3):
    if curve1.shape[0] < 2 or curve2.shape[0] < 2:
        raise ValueError('Number of function points must be at least 2')
    if curve1.shape[1] != 2 or curve2.shape[1] != 2:
        raise ValueError('Number of features must be 2')

    hx = np.mean(np.diff(curve1[:, 0]))
    distance = 0.
    n = 0
    for (index,), x0 in np.ndenumerate(curve1[:, 0]):
        if index < n_diff:
            [dx, dy] = curve1[index + n_diff] - curve1[index]
        elif index >= curve1.shape[0] - n_diff:
            [dx, dy] = curve1[index] - curve1[index - n_diff]
        else:
            [dx, dy] = curve1[index + n_diff] - curve1[index - n_diff]

        if fabs(dy) < eps:
            print("In x = {} the normal is directed strictly vertically".format(x0))
            difarray = np.abs(curve1[:, 0] - x0)
            index = np.argmin(difarray)
            if difarray[index] <= hx:
                p = curve1[index]
                distance += dist(curve1[index], p)
                n += 1

                continue
        if fabs(dx) < eps:
            print("In x = {} the normal is directed strictly horizontally".format(x0))
            continue

        k_norm = -1. / (dy / dx)
        z_norm = curve1[index, 1] + x0 / (dy / dx)

        if k_norm > 0:
            # Right
            # ЭТО ДЛЯ ФУНКЦИЙ ПОСТРОЕННЫХ НА ОДИНАКОВЫХ СЕТКАХ ПО Х, НАДО ИСПРАВИТЬ!
            x_min = x0
            x_max = curve2[-1, 0]

        else:  # k_line < 0
            # Left
            # ЭТО ДЛЯ ФУНКЦИЙ ПОСТРОЕННЫХ НА ОДИНАКОВЫХ СЕТКАХ ПО Х, НАДО ИСПРАВИТЬ
            x_min = curve2[0, 0]
            x_max = x0

        curve2range = curve2[np.logical_and(x_min <= curve2[:, 0], x_max >= curve2[:, 0])]

        try:
            p = intersection_line_curve(k_norm, z_norm, curve2range)
        except Exception:
            continue
        distance += dist(curve1[index], p)
        n += 1

    distance /= n
    return distance

def functions_distance_vertical(curve1, curve2):
    if curve1.shape[0] < 2 or curve2.shape[0] < 2:
        raise ValueError('Number of function points must be at least 2')
    if curve1.shape[1] != 2 or curve2.shape[1] != 2:
        raise ValueError('Number of features must be 2')

    mean = np.mean(np.abs(curve2[:, 1] - curve1[:, 1]))
    return mean
