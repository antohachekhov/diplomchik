import numpy as np
import cv2 as cv
from math import ceil, sqrt, log
from matplotlib import pyplot as plt
import matplotlib.patches as patches
from matplotlib.lines import Line2D

def method_prism(img, win_size):
    """
    Измерение фрактальных размерностей с помощью метода призм

    Метод основан на измерении площади поверхности призм, наложенных на поверхность при разных масштабах

    """
    t_win = img
    sub_windows_sizes = [win_size, int(win_size / 2), int(win_size / 4), int(win_size / 6)]

    # массив значений площади поверхности призм при разных размерах субокон
    Se = np.zeros(4)

    # проход по каждому размеру субокна
    for si in range(len(sub_windows_sizes)):
        size = sub_windows_sizes[si]

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
    lge = np.zeros(len(sub_windows_sizes))
    for k in range(len(sub_windows_sizes)):
        lgS[k] = log(Se[k])
        lge[k] = log(sub_windows_sizes[k])

    # по МНК находится наклон регрессии логарифма площади от логарифма размера субокна
    A = np.vstack([lge, np.ones(len(lge))]).T
    D, _ = np.linalg.lstsq(A, lgS, rcond=None)[0]

    # отрицательный наклон регрессии + 2 - есть фрактальная размерность в данной точке
    return 2 - D


img = cv.imread(r"C:\Users\bortn\Desktop\diplomchik\analysis\new dataset\20\1-04\1-04.tif", cv.IMREAD_GRAYSCALE)
img = img[:-103]

win_size = 60

coorTL = [463, 288]
coorM1 = [522, 153]
coorM2 = [464, 494]

trans_l = img[coorTL[1] - int(win_size / 2): coorTL[1] + int(win_size / 2) - 1, coorTL[0] - int(win_size / 2): coorTL[0] + int(win_size / 2) - 1]
m1 = img[coorM1[1] - int(win_size / 2): coorM1[1] + int(win_size / 2) - 1, coorM1[0] - int(win_size / 2): coorM1[0] + int(win_size / 2) - 1]
m2 = img[coorM2[1] - int(win_size / 2): coorM2[1] + int(win_size / 2) - 1, coorM2[0] - int(win_size / 2): coorM2[0] + int(win_size / 2) - 1]

# plt.imshow(img, cmap='gray')
# plt.scatter(coorTL[0], coorTL[1])
# plt.scatter(coorM1[0], coorM1[1])
# plt.scatter(coorM2[0], coorM2[1])
# plt.show()

FD = [0., 0., 0.]
for i, im in enumerate([trans_l, m1, m2]):
    # plt.imshow(im, cmap='gray')
    # plt.show()
    FD[i] = method_prism(im, win_size)

print(FD)

fig, ax = plt.subplots(1)
ax.imshow(img, cmap='gray')
rectTL = patches.Rectangle((coorTL[0] - win_size / 2, coorTL[1] - win_size / 2), win_size, win_size, linewidth=2,
                         edgecolor='w', facecolor="none")
rectM1 = patches.Rectangle((coorM1[0] - win_size / 2, coorM1[1] - win_size / 2), win_size, win_size, linewidth=2,
                         edgecolor='w', facecolor="none")
rectM2 = patches.Rectangle((coorM2[0] - win_size / 2, coorM2[1] - win_size / 2), win_size, win_size, linewidth=2,
                         edgecolor='w', facecolor="none")

lineTL1 = Line2D([coorTL[0] - win_size / 2 + 1, coorTL[0] + win_size / 2], [coorTL[1] - win_size / 2, coorTL[1] + win_size / 2 - 1], color="w", lw=0.5)
lineTL2 = Line2D([coorTL[0] + win_size / 2, coorTL[0] - win_size / 2 + 1], [coorTL[1] - win_size / 2, coorTL[1] + win_size / 2 - 1], color="w", lw=0.5)

lineM11 = Line2D([coorM1[0] - win_size / 2 + 1, coorM1[0] + win_size / 2], [coorM1[1] - win_size / 2, coorM1[1] + win_size / 2 - 1], color="w", lw=0.5)
lineM12 = Line2D([coorM1[0] + win_size / 2, coorM1[0] - win_size / 2 + 1], [coorM1[1] - win_size / 2, coorM1[1] + win_size / 2 - 1], color="w", lw=0.5)

lineM21 = Line2D([coorM2[0] - win_size / 2 + 1, coorM2[0] + win_size / 2], [coorM2[1] - win_size / 2, coorM2[1] + win_size / 2 - 1], color="w", lw=0.5)
lineM22 = Line2D([coorM2[0] + win_size / 2, coorM2[0] - win_size / 2 + 1], [coorM2[1] - win_size / 2, coorM2[1] + win_size / 2 - 1], color="w", lw=0.5)

ax.add_line(lineTL1)
ax.add_line(lineTL2)
ax.add_line(lineM11)
ax.add_line(lineM12)
ax.add_line(lineM21)
ax.add_line(lineM22)

rectTL_value = patches.Rectangle((coorTL[0] - win_size / 2, coorTL[1] - win_size / 2 - 15), win_size - 10, 15, linewidth=2,
                         edgecolor='w', facecolor="w")
rectM1_value = patches.Rectangle((coorM1[0] - win_size / 2, coorM1[1] - win_size / 2 - 15), win_size - 10, 15, linewidth=2,
                         edgecolor='w', facecolor="w")
rectM2_value = patches.Rectangle((coorM2[0] - win_size / 2, coorM2[1] - win_size / 2 - 15), win_size - 10, 15, linewidth=2,
                         edgecolor='w', facecolor="w")

ax.add_patch(rectTL)
ax.add_patch(rectM1)
ax.add_patch(rectM2)

ax.add_patch(rectTL_value)
ax.add_patch(rectM1_value)
ax.add_patch(rectM2_value)

ax.text(coorTL[0] - win_size / 2, coorTL[1] - win_size / 2 - 5, r'$D={:.2f}$'.format(FD[0]), fontsize=10)
ax.text(coorM1[0] - win_size / 2, coorM1[1] - win_size / 2 - 5, r'$D={:.2f}$'.format(FD[1]), fontsize=10)
ax.text(coorM2[0] - win_size / 2, coorM2[1] - win_size / 2 - 5, r'$D={:.2f}$'.format(FD[2]), fontsize=10)

plt.show()






