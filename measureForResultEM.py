import numpy as np
from sklearn.mixture import GaussianMixture
import cv2 as cv
import matplotlib.pyplot as plt
from math import fabs, sqrt
from FunctionsDistance import Line


def segment_distribution(img, intensity, win_size, show_step=True, dx=1, dy=1):
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
    radius = int(win_size / 2 if win_size / 2 % 2 == 1 else win_size / 2 - 1)
    kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE, (radius, radius))
    result = cv.dilate(img2, kernel)

    if show_step:
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


def segment_min_distribution(img, win_size, field, n_comp, dx=1, dy=1):
    """
    Сегментация переходного слоя через EM-классификацию

    :param n_comp: int
        Количество компонент в смеси распределений

    """
    # классификация значений поля по EM-алгоритму с тремя компонентами в смеси
    X = np.expand_dims(field.flatten(), 1)
    EMmodel = GaussianMixture(n_components=n_comp, covariance_type='full')
    EMmodel.fit(X)
    predict = EMmodel.predict(X)
    means = EMmodel.means_
    field_predict = np.resize(predict, field.shape)

    # перевод поля значений классов в пространство оттенков серого
    field_pr_img = field_predict / (n_comp - 1) * 255
    field_pr_img = field_pr_img.astype(np.uint8)

    # cv.imshow('EM', field_pr_img)

    # определение какому классу соответствуют значения, принадлежащие классу с минимальным математическим ожиданием
    stratum_class = np.argmin(means)
    intensive_stratum = int(stratum_class / (n_comp - 1) * 255)

    # выделение областей, принадлежащих классу с минимальным математическим ожиданием
    # получаем маску
    mask_stratum = segment_distribution(field_pr_img, intensive_stratum, win_size)

    # переносим получившуюся маску поля на маску для изображения
    mask_stratum = cv.resize(mask_stratum,
                             np.flip((img.shape[0] - win_size + 1,
                                      img.shape[1] - win_size + 1)),
                             fx=dx, fy=dy, interpolation=cv.INTER_LINEAR_EXACT)
    segments = np.zeros(img.shape, dtype=np.uint8)
    segments[int(win_size / 2): -int(win_size / 2) + 1, int(win_size / 2): -int(win_size / 2) + 1] = mask_stratum
    return segments


def distance(point1, point2):
    """
    Вычисление расстояния между двумя точками
    :param point1: тип list - (x,y)
    :param point2: тип list - (x,y)
    :return:
    """
    return sqrt((point1[0] - point2[0]) ** 2 + (point1[1] - point2[1]) ** 2)

hsv_min = 254
hsv_max = 255

if __name__ == '__main__':
    # Тест с фигурами

    # круг - 600
    #image = cv.imread(r"C:\Users\bortn\Desktop\diplomchik\testMeasure2\circle600.jpg", cv.IMREAD_GRAYSCALE)

    # квадрат - 500
    # image = cv.imread(r"C:\Users\bortn\Desktop\diplomchik\testMeasure2\square500.jpg", cv.IMREAD_GRAYSCALE)

    # прямоугольник - 100
    # image = cv.imread(r"C:\Users\bortn\Desktop\diplomchik\testMeasure2\rectangle100.jpg", cv.IMREAD_GRAYSCALE)

    # прямоугольник под углом - 100
    #image = cv.imread(r"C:\Users\bortn\Desktop\diplomchik\testMeasure2\rectangleAngle100.jpg", cv.IMREAD_GRAYSCALE)

    # thresh = cv.inRange(image, hsv_min, hsv_max)
    # contours0, hierarchy = cv.findContours(thresh, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_NONE)
    # win_size = 0

    # Тест с переходной областью
    image = cv.imread(r"C:\Users\bortn\Desktop\diplomchik\analysis\old dataset\20min_1\[Filtered] 20.jpg",
                     cv.IMREAD_GRAYSCALE)
    field = np.loadtxt(r"C:\Users\bortn\Desktop\diplomchik\analysis\old dataset\20min_1\field_filtered_x1y1w30.csv",
                      delimiter=';')
    win_size = 30
    mask = segment_min_distribution(image, win_size, field, 3)
    nb_components, output, stats, centroids = cv.connectedComponentsWithStats(mask, 4, cv.CV_32S)
    contours0, hierarchy = cv.findContours(mask, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)

    index = 0
    layer = 0

    def update():
        vis = image.copy()
        cv.drawContours(vis, contours0, index, (150, 0, 0), 2, cv.LINE_AA, hierarchy, layer)
        cv.imshow('contours', vis)

    def update_index(v):
        global index
        index = v-1
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
        areaX = np.logical_and(contourObject[:, :, 0] > win_size / 2,
                               contourObject[:, :, 0] < image.shape[0] - win_size / 2)
        areaY = np.logical_and(contourObject[:, :, 1] > win_size / 2,
                               contourObject[:, :, 1] < image.shape[1] - win_size / 2)
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
                        testLeftPoint = cv.pointPolygonTest(contourObjectFiltered, (float(x0 - 1), norm(x0 - i)), False)
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
                                testLeftPoint = cv.pointPolygonTest(contourObjectFiltered, (float(x0 - i), norm(x0 - i)), False)
                                if cv.pointPolygonTest(contourObjectFiltered, (float(x0 + i), norm(x0 + i)), False) >= 0:
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
                                testLeftPoint = cv.pointPolygonTest(contourObjectFiltered, (float(x0 - i), norm(x0 - i)), False)
                                if cv.pointPolygonTest(contourObjectFiltered, (float(x0 + i), norm(x0 + i)), False) >= 0:
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
                    IndexPointsInDx = np.logical_and(contourObjectFiltered[:, 0] > Dx[0], contourObjectFiltered[:, 0] < Dx[1])
                    IndexPointsInDy = np.logical_and(contourObjectFiltered[:, 1] > Dy[0], contourObjectFiltered[:, 1] < Dy[1])
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
                    dInPoint = distance(np.array([x0, y0]), point)
                    if dInPoint < minDistance:
                        minDistance = dInPoint
                        minPoints = point
                if minDistance < 1e+20:
                    distances.append(minDistance)
                #print(minDistance)
            except:
                continue

    print(np.median(distances))
    plt.hist(distances)
    plt.show()

    # !!!!!!!!! ОТКУДА БЕРУТСЯ ЗНАЧЕНИЯ 300 ??????????7
    # 6.7802734375e-08 * 69.11410078031139