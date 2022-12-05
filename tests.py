from PIL import Image
import numpy as np



image_file = Image.open(r"C:\Users\bortn\Desktop\Diplomchichek\dataset\1100_РЭМ\5\1-15.tif").convert('L')
imgorig = np.asarray(image_file)

n = 2
start = 21
i = 0
while imgorig[700, start + i + 1] == 255:
    i += 1
n += i + 3
print(n)


"""
measurement_dict = {
    "m": 1E-3,
    "µ": 1E-6,
    "n": 1E-9,
}

img_file = Image.open(r"")
exif_data = img_file.getexif()

tags = exif_data.items()._mapping[34118]

width = 0.

for row in tags.split("\r\n"):
    key_value = row.split(" = ")
    if len(key_value) == 2:
        if key_value[0] == "Width":
            width = key_value[1]
            break

print(width)

width_str = width.split(" ")

width_unit = width_str[1]
width_value = float(width_str[0]) * measurement_dict[width_unit[0]]


print(width_value, 'm')

print('Done')




def f_kas(fx0, dfx0, x0, x):
    kas = x.copy()
    kas -= x0
    kas *= dfx0
    kas += fx0
    return kas


def f_norm(fx0, dfx0, x0, x):
    norm = x.copy()
    norm -= x0
    norm *= 1. / (dfx0)
    norm = fx0 - norm
    return norm


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

        if math.fabs(dy) < eps:
            print("In x = {} the normal is directed strictly vertically".format(x0))
            difarray = np.abs(curve1[:, 0] - x0)
            index = np.argmin(difarray)
            if difarray[index] <= hx:
                p = curve1[index]
                distance += math.dist(curve1[index], p)
                n += 1

                continue
        if math.fabs(dx) < eps:
            print("In x = {} the normal is directed strictly horizontally".format(x0))
            continue

        k_norm = -1. / (dy / dx)
        z_norm = curve1[index, 1] + x0 / (dy / dx)

        if k_norm > 0:
            # Right
            x_min = x0
            x_max = curve2[-1, 0]

        else:  # k_line < 0
            # Left
            x_min = curve2[0, 0]
            x_max = x0

        curve2range = curve2[np.logical_and(x_min <= x, x_max >= x)]

        try:
            p = intersection_line_curve(k_norm, z_norm, curve2range)
        except Exception:
            continue
        distance += math.dist(curve1[index], p)
        n += 1

    distance /= n
    return distance


# if index % 10 == 0:
#     plt.plot(curve1[:, 0], curve1[:, 1])
#     plt.plot(curve2[:, 0], curve2[:, 1])
#     plt.plot([curve1[index, 0], p[0]], [curve1[index, 1], p[1]])
#     plt.scatter(p[0], p[1])
#     plt.scatter(curve1[index, 0], curve1[index, 1])
#     plt.show()



# x = -1.4181818
# x0 = -6
# normal = Line(a=-1./-1.2, c=3.6+x0/-1.2)
# print(normal.y(x0))

def myfunc(x):
    y = x**2
    return y


error = False
points = False
inclination = False


eps = 0.0000001
h = 0.1
x = np.arange(-7, 7 + h, h)
f = np.vectorize(math.sin)
# f = np.vectorize(myfunc)
y1 = f(x)
f2 = np.vectorize(math.sin)
y2 = f2(x) + 5
y1array = np.array([x, y1]).T
y2array = np.array([x, y2]).T

fig, ax = plt.subplots(1)
ax.plot(y1array[:, 0], y1array[:, 1])
ax.plot(y2array[:, 0], y2array[:, 1])
ax.xaxis.set_major_locator(ticker.MultipleLocator(0.5))
ax.xaxis.set_minor_locator(ticker.MultipleLocator(0.1))
ax.yaxis.set_major_locator(ticker.MultipleLocator(0.5))
ax.yaxis.set_minor_locator(ticker.MultipleLocator(0.1))
ax.grid(which='major')
ax.grid(which='minor')
plt.show()

ind = 80
x0 = x[ind]
n = 3

d = functions_distance(y1array, y2array)
print(d)


dy = y1[ind + n] - y1[ind - n]
if math.fabs(dy) < eps:
    # dy = 0.
    # нормаль направленна вверх
    difarray = np.abs(y2array[:, 0] - x0)
    index = np.argmin(difarray)
    if difarray[index] <= 2 * h:
        p = y2array[index]
        print(p)
else:

    inclination = True
    left = False

    dx = h * n * 2
    dfx0 = dy / dx
    fx0 = y1[ind]
    # Method 1
    k_norm = -1. / dfx0
    z_norm = fx0 + x0 / dfx0
    x_min = 0
    x_max = 0

    p = np.array([0, 0])

    if k_norm > 0:
        x_min = x0
        x_max = y2array[-1, 0]
        left = False

    elif k_norm < 0:
        x_min = y2array[0, 0]
        x_max = x0
        left = True

    else:
        print('zero')
        error = True

        # ДОБАВИТЬ ПОТОМ


    if left:
        sign_direction = -1
        y_direction = 1
    else:
        sign_direction = 0
        y_direction = -1

    direction = -1 if left else 0

    xrange = np.logical_and(x_min <= x, x_max >= x)
    y2_in_range = y2array[xrange]

    normalModel = Line(a=k_norm, c=z_norm)
    eq = normalModel.kanon_eq_vector(y2_in_range)
    eq[np.abs(eq) < eps] = 0.
    eq_sign = np.sign(eq)

    if np.all(eq_sign == eq_sign[0]):
        print('Пересечения нет')
        error = True
    else:
        eq_signchange = ((np.roll(eq_sign, y_direction) - eq_sign) != 0).astype(int)
        change_index = np.where(eq_signchange == 1)[0]

        if eq_sign[change_index[sign_direction]] == 0:
            p = y2_in_range[change_index[sign_direction]]
            print(p)
        else:
            a = y2_in_range[change_index[sign_direction] - y_direction]
            b = y2_in_range[change_index[sign_direction]]
            points = True

            k_segment = (b[1] - a[1]) / (b[0] - a[0])
            z_segment = (b[1] - a[1]) * (-a[0] / (b[0] - a[0]) + a[1] / (b[1] - a[1]))

            px = (z_segment - z_norm) / (k_norm - k_segment)
            py = (k_norm * z_segment - k_segment * z_norm) / (k_norm - k_segment)
            p = np.array([px, py])

            print(p)

if not error:
    fig, ax = plt.subplots(1)

    if inclination:
        y_min = np.amin(y2array[xrange, 1])
        y_max = np.amax(y2array[xrange, 1])
        normal = np.array([x[xrange], normalModel.y(x[xrange])]).T
        ax.plot(y2_in_range[:, 0], y2_in_range[:, 1])
        ax.plot(normal[:, 0], normal[:, 1])
        ax.scatter(y2array[:, 0], y2array[:, 1], color='black', s=4)
    else:
        ax.plot(y2array[:, 0], y2array[:, 1])
    #normalrange = np.logical_and(y_min - math.fabs(y_max - y_min) <= normal[:, 1], math.fabs(y_max - y_min) + y_max >= normal[:, 1])
    #normal = normal[normalrange]

    ax.xaxis.set_major_locator(ticker.MultipleLocator(0.5))
    ax.xaxis.set_minor_locator(ticker.MultipleLocator(0.1))
    ax.yaxis.set_major_locator(ticker.MultipleLocator(0.5))
    ax.yaxis.set_minor_locator(ticker.MultipleLocator(0.1))
    ax.grid(which='major')
    ax.grid(which='minor')

    ax.plot(x, y1)


    if points:
        ax.scatter(a[0], a[1])
        ax.scatter(b[0], b[1])

    ax.scatter(p[0], p[1])

    plt.show()
    # normal = np.array([x[xrange], normalModel.y(x[xrange])]).T
    # normalrange = np.logical_and(y_min - math.fabs(y_max - y_min) <= normal[:, 1], math.fabs(y_max - y_min) + y_max >= normal[:, 1])
    # normal = normal[normalrange]

    # if True in np.logical_and(y_min <= normal[:, 1], y_max >= normal[:, 1]):

    
        nbrs = NearestNeighbors(n_neighbors=1,
                                radius=0.1,
                                algorithm='auto',
                                metric='euclidean').fit(normal)

        distances, _ = nbrs.kneighbors(y2_in_range)
        # distances, indices = nbrs.kneighbors(y2_in_range)
        a_idx = np.argmin(distances, axis=0)
        a = y2_in_range[a_idx][0]
        res = normalModel.kanon_eq(a)
        

        if res == 0:
            p = a
        else:
            if res > 0:
                y2_half_plane = y2_in_range[normalModel.kanon_eq_vector(y2_in_range) < 0]

            else:
                y2_half_plane = y2_in_range[normalModel.kanon_eq_vector(y2_in_range) > 0]

            distances, _ = nbrs.kneighbors(y2_half_plane)
            b_idx = np.argmin(distances, axis=0)
            b = y2_half_plane[b_idx][0]

            k_segment = (b[1] - a[1]) / (b[0] - a[0])
            z_segment = (b[1] - a[1]) * (-a[0] / (b[0] - a[0]) + a[1] / (b[1] - a[1]))

            px = (z_segment - z_norm) / (k_norm - k_segment)
            py = (k_norm * z_segment - k_segment * z_norm) / (k_norm - k_segment)

            #px = (a[1] - a[0] * (b[1] - a[1]) / (b[0] - a[0]) + fx0 + x0 / dfx0) / \
            #     (-1./dfx0 - (b[1] - a[1]) / (b[0] - a[0]))

            #py = ((b[1] - a[1]) / (b[0] - a[0]) * ((a[0] + x0) / dfx0 + fx0) - a[1] / dfx0) / \
            #     (-1./dfx0 - (b[1] - a[1]) / (b[0] - a[0]))

            p = np.array([px, py])

            # if y2_half_plane.shape[0] == 0:
                # бряк

        ax.xaxis.set_major_locator(ticker.MultipleLocator(0.5))
        ax.xaxis.set_minor_locator(ticker.MultipleLocator(0.1))
        ax.yaxis.set_major_locator(ticker.MultipleLocator(0.5))
        ax.yaxis.set_minor_locator(ticker.MultipleLocator(0.1))
        ax.grid(which='major')
        ax.grid(which='minor')

        ax.plot(x, y1)
        ax.plot(y2_in_range[:, 0], y2_in_range[:, 1])
        ax.plot(normal[:, 0], normal[:, 1])

        ax.scatter(a[0], a[1])
        ax.scatter(b[0], b[1])
        ax.scatter(p[0], p[1])
        ax.scatter(y2array[:, 0], y2array[:, 1], color='black', s=4)

        plt.show()


# else:

norm_eq = Line(a=-1. / dfx0, c=fx0 + x0 / dfx0)
norm_vector = norm_eq.y(x)

kas = f_kas(fx0, dfx0, x0, x)
norm = f_norm(fx0, dfx0, x0, x)

print('kas-k = ', dfx0)
print('norm-k = ', -1. / dfx0)
print('kas-k - norm-k = ', dfx0 + 1. / dfx0)
print('kas-k * norm-k = ', dfx0 * (-1. / dfx0))
print('tg(gamma) = ', (dfx0 + 1. / dfx0) / (1 + dfx0 * (-1. / dfx0)))
print('gamma', math.atan((dfx0 + 1. / dfx0) / (1 + dfx0 * (-1. / dfx0))))
print('gamma in degrees = ', math.degrees(math.atan((dfx0 + 1. / dfx0) / (1 + dfx0 * (-1. / dfx0)))))

points_n = np.array([x, norm]).T

n_neighbors = 1  # сколько ближайших соседей хотим найти
nbrs = NearestNeighbors(n_neighbors=n_neighbors,
                        radius=0.1,
                        algorithm='auto',
                        metric='euclidean').fit(points_n)

points_y2 = np.array([x, y2]).T

distances, indices = nbrs.kneighbors(points_y2)
a_idx = np.argmin(distances, axis=0)
p = points_n[indices[a_idx][0][0]]

#
#for i, idx in enumerate(a_idx):
#    distance = distances[idx][i]
#    p1 = points_n[indices[idx][i]]
#    p2 = points_y2[idx]
#    print("Наименьшее расстояние между точками {} и {}, расстояние равно {}".format(p1, p2, distance))
#    ax.scatter(p1[0], p1[1])
#    ax.scatter(p2[0], p2[1])


ax.xaxis.set_major_locator(ticker.MultipleLocator(0.5))
ax.xaxis.set_minor_locator(ticker.MultipleLocator(0.1))
ax.yaxis.set_major_locator(ticker.MultipleLocator(0.5))
ax.yaxis.set_minor_locator(ticker.MultipleLocator(0.1))
ax.grid()

ax.plot(x, y1)
ax.plot(x, y2)
ax.plot(x, kas)
ax.plot(x, norm)
ax.scatter(p[0], p[1])
ax.scatter(x[ind], y1[ind])

print('distance = ', math.sqrt((p[0] - x[ind]) ** 2 + (p[1] - y1[ind]) ** 2))

'''
unit = 0.5
x_tick = np.arange(-0.1, 0.1+unit, unit)
x_label = [r"$-\frac{\pi}{2}$", r"$-\frac{\pi}{4}$", r"$0$", r"$+\frac{\pi}{4}$",   r"$+\frac{\pi}{2}$"]
ax.set_xticks(x_tick*np.pi)
ax.set_xticklabels(x_label, fontsize=12)
'''

plt.show()

"""
