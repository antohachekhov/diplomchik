import numpy as np
import cv2 as cv
from math import ceil, sqrt, log
import matplotlib.pyplot as plt
from numba import njit, prange
import multiprocessing
import time

def generate_windows(img, win_size, dx, dy):
    """
    Генерация измерительных окон в каждой точке изображения

    """
    # win = np.zeros((self._win_size, self._win_size))
    # вычисляется размер массива измерительных окон
    shape = img.shape[:-2] + ((img.shape[-2] - win_size) // dy + 1,) + \
            ((img.shape[-1] - win_size) // dx + 1,) + (win_size, win_size)
    # вычисляется количество шагов измерительного окна
    strides = img.strides[:-2] + (img.strides[-2] * dy,) + \
              (img.strides[-1] * dx,) + img.strides[-2:]
    return np.lib.stride_tricks.as_strided(img, shape=shape, strides=strides)


def method_prism(windows, sub_windows_sizes):
    """
    Измерение фрактальных размерностей с помощью метода призм

    Метод основан на измерении площади поверхности призм, наложенных на поверхность при разных масштабах

    """
    field = np.zeros((windows.shape[0], windows.shape[1]))

    # проход по каждому измерительному окну
    for i, windows_str in enumerate(windows):
        for j, t_win in enumerate(windows_str):

            # массив значений площади поверхности призм при разных размерах субокон
            Se = np.zeros(sub_windows_sizes.shape)

            # проход по каждому размеру субокна
            for si, size in enumerate(sub_windows_sizes):
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
            lge = np.zeros(sub_windows_sizes.shape)
            for k in range(sub_windows_sizes.shape[0]):
                lgS[k] = log(Se[k])
                lge[k] = log(sub_windows_sizes[k])

            # по МНК находится наклон регрессии логарифма площади от логарифма размера субокна
            A = np.vstack([lge, np.ones(len(lge))]).T
            D, _ = np.linalg.lstsq(A, lgS, rcond=None)[0]

            # отрицательный наклон регрессии + 2 - есть фрактальная размерность в данной точке
            field[i][j] = 2 - D

        print('\rMethod of prism [', '█' * int(i / windows.shape[0] * 20),
              ' ' * int((windows.shape[0] - i) / windows.shape[0] * 20), ']\t', i, '\t/ ',
              windows.shape[0] - 1,
              end='')
    print('\n')

    return field


def method_prism_parallel(windows, sub_windows_sizes):
    """
    Измерение фрактальных размерностей с помощью метода призм

    Метод основан на измерении площади поверхности призм, наложенных на поверхность при разных масштабах

    """
    field = np.zeros(windows.shape[0])

    # проход по каждому измерительному окну
    for j in range(windows.shape[0]):
        t_win = windows[j]

        # массив значений площади поверхности призм при разных размерах субокон
        Se = np.zeros(sub_windows_sizes.shape)

        # проход по каждому размеру субокна
        for si, size in enumerate(sub_windows_sizes):
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
        lge = np.zeros(sub_windows_sizes.shape)
        for k in range(sub_windows_sizes.shape[0]):
            lgS[k] = log(Se[k])
            lge[k] = log(sub_windows_sizes[k])

        # по МНК находится наклон регрессии логарифма площади от логарифма размера субокна
        A = np.vstack([lge, np.ones(len(lge))]).T
        D, _ = np.linalg.lstsq(A, lgS, rcond=None)[0]

        # отрицательный наклон регрессии + 2 - есть фрактальная размерность в данной точке
        field[j] = 2 - D

        #print('\rMethod of prism [', '█' * int(i / windows.shape[0] * 20),
        #      ' ' * int((windows.shape[0] - i) / windows.shape[0] * 20), ']\t', i, '\t/ ',
        #      windows.shape[0] - 1,
        #      end='')
    #print('\n')

    return field

def field_to_image(input_field, show=False):
    """
    Перевод поля фрактальных размерностей в пространство оттенков серого

    :param input_field: 2-D numpy массив
        Поле фрактальных размерностей.
    :param show: bool
        True, если необходимо вывести изображение визуализации поля

    :return img_out: 2-D numpy массив
        Изображение-визуализация поля фрактальных размерностей
    """
    img_out = np.empty((input_field.shape[0], input_field.shape[1]), dtype=np.uint8)
    for i in range(img_out.shape[0]):
        for j in range(img_out.shape[1]):

            if input_field[i][j] - 1.0 > 0.0:
                img_out[i][j] = int(255 * (input_field[i][j] - 1) / 3)
            else:
                img_out[i][j] = 0
    if show:
        plt.imshow(img_out, cmap='gray')
        plt.show()
    return img_out


if __name__ == '__main__':
    win_size = 32
    sub_windows_sizes = np.array([win_size,
                                  int((win_size + 1) / 2),
                                  int((win_size + 3) / 4)])
    image = cv.imread(r"C:\Users\bortn\Desktop\diplomchik\analysis\old dataset\5min_1\[Filtered] test.jpg", cv.IMREAD_GRAYSCALE)
    windows = generate_windows(image, win_size, 1, 1)
    # p = multiprocessing.Process(target=method_prism, args=(windows, sub_windows_sizes, ))
    # p.start()
    # p.join()
    windows1 = windows[:, :, :, :]

    # Тест для параллельного
    timer = time.time()
    parallel_result = list()
    with multiprocessing.Pool() as pool:
        print('Result: ')
        results = [pool.apply_async(method_prism_parallel, args=(arg1, sub_windows_sizes)) for arg1 in windows1]

        for res in results:
            parallel_result.append(res.get())

        pool.close()
        pool.join()

    parallel_result = np.asarray(parallel_result)
    print('Тест с параллельными вычислениями занял %.6f' % (time.time() - timer))

    # Тест без параллельного
    timer = time.time()
    notParallelResult = method_prism(windows1, sub_windows_sizes)
    print('Тест без параллельных вычислений занял %.6f' % (time.time() - timer))

    imgParallel = field_to_image(parallel_result, False)
    imgNotParallel = field_to_image(notParallelResult)

    fig, axis = plt.subplots(1, 2)
    axis[0].imshow(imgNotParallel, cmap='gray')
    axis[0].set_title('Не параллельно')
    axis[1].imshow(imgParallel, cmap='gray')
    axis[1].set_title('Параллельно')
    plt.show()
    print('end')
