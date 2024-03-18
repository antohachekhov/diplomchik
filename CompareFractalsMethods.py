from parallelPrism import method_prism_parallel, generate_windows, field_to_image
import cv2 as cv
from math import ceil, log, sqrt
import numpy as np
import multiprocessing
import matplotlib.pyplot as plt
from sklearn.mixture import GaussianMixture

testFieldCSV = False
imageFile = r"C:\Users\bortn\Desktop\diplomchik\analysis\old dataset\20min_1\20_1.jpg"


def method_cubes_parallel(windows, sub_windows_sizes):
    """
    Измерение фрактальных размерностей с помощью метода кубов

    Метод основан на подсчёте количества кубов, необходимых для полного покрытия поверхности на разных масштабах

    """
    field = np.zeros(windows.shape[0])

    # проход по каждому измерительному окну
    for j in range(windows.shape[0]):
        t_win = windows[j]

        # массив количества кубов для разных размеров субокон
        n = np.zeros(sub_windows_sizes.shape)

        # проход по каждому размеру вспомогательных окон
        for ei in range(sub_windows_sizes.shape[0]):
            size = sub_windows_sizes[ei]

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
        lge = np.zeros(sub_windows_sizes.shape)
        for k in range(sub_windows_sizes.shape[0]):
            lgn[k] = log(n[k])
            lge[k] = log(sub_windows_sizes[k])

        # по МНК находим наклон регрессии логарифма количества кубов от логарифма размера субокон
        A = np.vstack([lge, np.ones(len(lge))]).T
        D, _ = np.linalg.lstsq(A, lgn, rcond=None)[0]

        # отрицательная величина наклона регрессии - величина фрактальной размерности в данной точке
        field[j] = -D

    return field


def EMAnalysis(field, comp):
    X = np.expand_dims(field.flatten(), 1)
    model = GaussianMixture(n_components=comp, covariance_type='full')
    model.fit(X)
    mu = model.means_
    sigma = list(map(sqrt, model.covariances_))
    for i in range(comp):
        print(f"mu_{i} = {mu[i,0]:.4f}    sigma_{i} = {sigma[i]:.4f}")


def showResults(field):
    plt.grid(True)
    counts, bins, _ = plt.hist(field.flatten(), bins=250, facecolor='green',
                               edgecolor='none', density=True, label='ПФР')
    plt.xlabel("Фрактальная размерность, $\it{D}$")
    plt.ylabel('Частота')
    plt.show()

    imgParallel = field_to_image(field, False)
    plt.imshow(imgParallel, cmap='gray')
    plt.show()


methods = {'1': method_cubes_parallel,
           '2': method_prism_parallel}

if __name__ == '__main__':
    winSize = int(input("Введите размер окна: "))
    dx, dy = map(int, input("Введите dx и dy через пробел: ").split())
    if not testFieldCSV:
        methodKey = input("1 - метод кубов, 2 - метод призм: ")
        image = cv.imread(imageFile, cv.IMREAD_GRAYSCALE)
        windows = generate_windows(image, winSize, dx, dy)
        sub_windows_sizes = np.array([winSize,
                                      int((winSize + 1) / 2),
                                      int((winSize + 3) / 4),
                                      int((winSize + 7) / 8)])

        parallel_result = list()

        with multiprocessing.Pool() as pool:
            print('Обработка началась...')
            results = [pool.apply_async(methods[methodKey], args=(arg1, sub_windows_sizes)) for arg1 in windows]

            for res in results:
                parallel_result.append(res.get())

            pool.close()
            pool.join()

        print('Обработка закончилась')

        parallel_result = np.asarray(parallel_result)

        fileName = input("Введите название файла для сохранения: ")
        try:
            np.savetxt(r'./fieldsForCompare/' + fileName + '.csv', parallel_result, delimiter=";", newline="\n")
        except Exception as e:
            print("Ошибка сохранения")
    else:
        parallel_result = np.loadtxt(r"C:\Users\bortn\Desktop\diplomchik\analysis\old dataset\5min_1\field_filteredclassic_x1y1w30.csv", delimiter=";")

    #fig, ax = plt.subplot()
    showResults(parallel_result)
    comp = int(input("Введите количество компонент: "))
    EMAnalysis(parallel_result, comp)

    print('end')
