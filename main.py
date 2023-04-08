import math
import os.path
import cv2 as cv
import numpy as np
import matplotlib.lines as lns
import tkinter.messagebox as mb
import matplotlib.pyplot as plt

from PIL import Image
from tkinter import Tk
from tkinter.filedialog import askopenfilename
from FractalAnalysisClass import FractalAnalysis

from sklearn.mixture import GaussianMixture
import scipy.stats as stats

Test = True

nametest = '5min_1'

tests = {
    '5min_1': {'image': 'C:/Users/bortn/Desktop/diplomchik/analysis/old dataset/5min_1/[Filtered] test.jpg',
               'filtered': True,
               'field': r'C:\Users\bortn\Desktop\diplomchik\analysis\old dataset\5min_1\field_filteredclassic_x1y1w30.csv',
               'win_size': 30,
               'dx': 1,
               'dy': 1},
    '5min_2': {'image': r'C:\Users\bortn\Desktop\diplomchik\analysis\old dataset\5min_2\[Filtered] 5_2.jpg',
               'filtered': True,
               'field': r'C:\Users\bortn\Desktop\diplomchik\analysis\old dataset\5min_2\field_filtered_x1y1w30.csv',
               'win_size': 30,
               'dx': 1,
               'dy': 1},
    '10min_1': {'image': r"C:\Users\bortn\Desktop\diplomchik\analysis\old dataset\10min_1\[Filtered] 10_1.jpg",
                'filtered': True,
                'field': r"C:\Users\bortn\Desktop\diplomchik\analysis\old dataset\10min_1\field_filtered_50-175_x1y1w30.csv",
                'win_size': 30,
                'dx': 1,
                'dy': 1},
    '10min_2': {'image': r"C:\Users\bortn\Desktop\diplomchik\analysis\old dataset\10min_2\[Filtered] 10_2 50_150.jpg",
                'filtered': True,
                'field': r"C:\Users\bortn\Desktop\diplomchik\analysis\old dataset\10min_2\field_filt50_150_w30x1y1.csv",
                'win_size': 30,
                'dx': 1,
                'dy': 1},
    '15min_1': {'image': r"C:\Users\bortn\Desktop\diplomchik\analysis\old dataset\15min_1\[Filtered] 15_1.jpg",
                'filtered': True,
                'field': r"C:\Users\bortn\Desktop\diplomchik\analysis\old dataset\15min_1\field_filtered_w30x1y1.csv",
                'win_size': 30,
                'dx': 1,
                'dy': 1},
    '15min_2': {'image': r"C:\Users\bortn\Desktop\diplomchik\analysis\old dataset\15min_2\[Filtered] 15_2.jpg",
                'filtered': True,
                'field': r"C:\Users\bortn\Desktop\diplomchik\analysis\old dataset\15min_2\field_filtered_x1y1w30.csv",
                'win_size': 30,
                'dx': 1,
                'dy': 1},
    '20min_1': {'image': 'C:/Users/bortn\Desktop/diplomchik/analysis/old dataset/20min_1/[Filtered] 20_1.jpg',
                'filtered': True,
                'field': 'C:/Users/bortn/Desktop/diplomchik/analysis/old dataset/20min_1/field_filtered_x1y1w30.csv',
                'win_size': 30,
                'dx': 1,
                'dy': 1
                },
    '20min_2': {'image': r"C:\Users\bortn\Desktop\diplomchik\analysis\old dataset\20min2\[Filtered] 20-2.jpg",
                'filtered': True,
                'field': r"C:\Users\bortn\Desktop\diplomchik\analysis\old dataset\20min2\field_filtered_x1y1w30.csv",
                'win_size': 30,
                'dx': 1,
                'dy': 1
                },
    '25min_1': {'image': r"C:\Users\bortn\Desktop\diplomchik\analysis\old dataset\25min_1\[Filtered] 25_1.jpg",
                'filtered': True,
                'field': r"C:\Users\bortn\Desktop\diplomchik\analysis\old dataset\25min_1\field_filtered_x1y1w30.csv",
                'win_size': 30,
                'dx': 1,
                'dy': 1
                },
    '25min_2': {'image': r"C:\Users\bortn\Desktop\diplomchik\analysis\old dataset\25min_2\[Filtered] 25_2.jpg",
                'filtered': True,
                'field': r"C:\Users\bortn\Desktop\diplomchik\analysis\old dataset\25min_2\field_filtered_x1y1w30.csv",
                'win_size': 30,
                'dx': 1,
                'dy': 1
                },
}

colors = {'0': '#BEF73E',
          '1': '#C9007A',
          '2': '#FF7F00',
          '3': '#0772A1',
          '4': '#2818B1'}


def em_analysis(field):
    chisquares = np.zeros(shape=4)
    j = 0
    for comp in [2, 3, 4, 5]:
        X = np.expand_dims(field.flatten(), 1)
        model = GaussianMixture(n_components=comp, covariance_type='full')
        model.fit(X)
        mu = model.means_
        sigma = model.covariances_
        # print('Веса')
        # print(model.weights_)
        # print(mu)
        # print(sigma)

        # вывод гистограммы распределений фрактальных размерностей в поле
        # count - кол-во, bins - значение размерности
        counts, bins, _ = plt.hist(fractal.field.flatten(), bins=250, facecolor=colors['0'], edgecolor='none', density=True)
        np.savetxt(folder + '/' + 'counts' + '.csv', counts, delimiter=";", newline="\n")
        np.savetxt(folder + '/' + 'bins' + '.csv', bins, delimiter=";", newline="\n")
        plt.title('Кластеризация EM-алгоритмом по ' + str(comp) + ' распределениям')
        # for i in range(comp):
        #     xi = np.linspace(mu[i][0] - 5 * math.sqrt(sigma[i][0][0]), mu[i][0] + 5 * math.sqrt(sigma[i][0][0]), 100)
        #     plt.plot(xi, stats.norm.pdf(xi, mu[i][0], math.sqrt(sigma[i][0][0])),
        #              label=rf'$\mu_{i}={mu[i][0]:.6f}, \sigma_{i}$={sigma[i][0][0]:.6f}', linewidth=2)

        # xi = np.linspace(np.min(bins), np.max(bins), bins.shape[0])
        x = (bins[1:] + bins[:-1]) / 2.
        res = np.zeros(x.shape)
        for i in range(comp):
            res += model.weights_[i] * stats.norm.pdf(x, mu[i][0], math.sqrt(sigma[i][0][0]))
        plt.plot(x, res, color='red')
        plt.tight_layout()
        plt.legend()
        plt.grid(True)
        plt.show()

        chi, p = stats.chisquare(res, f_exp=counts)
        print('Значение для n_comp = {} Хи-квадрат = {}, p = {}'.format(comp, chi, p))
        chisquares[j] = chi
        j += 1

    print(chisquares)


def noise(img_gray, lower, upper, plot=False):
    """
    Функция избавляется от шумов на изображении (чёрные или белые пятна)

    :param img_gray: 3-D numpy массив.
        Изображение в цветовом пространстве оттенков серого.
    :param lower: Int
        Нижний предел допустимой (не шумовой) интенсивности цвета.
    :param upper: Int
        Верхний предел допустимой (не шумовой) интенсивности цвета.
    :param plot: bool.
        True, если необходимо вывести пошаговую обработку изображения.
        False, если это не нужно.

    :return: 3-D numpy массив размерности NxMx3.
        Обработанное изображение в цветовом пространстве BGR.
    """
    # перевод изображения в пространство оттенков серого
    # gray = cv.cvtColor(img_orig, cv.COLOR_BGR2GRAY)

    # Пороговая бинаризация изображение
    # пиксели, интенсивность которых не входит в диапазон [lower, upper], становятся чёрными. Остальные - белыми
    mask1 = cv.threshold(img_gray, lower, 255, cv.THRESH_BINARY_INV)[1]
    mask2 = cv.threshold(img_gray, upper, 255, cv.THRESH_BINARY)[1]
    mask = cv.add(mask1, mask2)
    mask = cv.bitwise_not(mask)

    # Увеличиваем радиус каждого чёрного объекта
    kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE, (5, 5))
    mask = cv.morphologyEx(mask, cv.MORPH_ERODE, kernel)

    # Делаем маску трёх-канальной
    # mask = cv.merge([mask, mask, mask])

    # Инвертируем маску
    mask_inv = 255 - mask

    # Считаем радиус для размытия областей с шумом
    radius = int(img_gray.shape[0] / 3) + 1
    if radius % 2 == 0:
        radius = radius + 1

    # Размываем изображение
    median = cv.medianBlur(img_gray, radius)

    # Накладываем маску на изображение
    img_masked = cv.bitwise_and(img_gray, mask)

    # Накладываем инвертированную маску на размытое изображение
    median_masked = cv.bitwise_and(median, mask_inv)

    # Соединяем два полученных изображения
    result = cv.add(img_masked, median_masked)

    if plot:
        plt.subplot(2, 3, 1)
        plt.imshow(img_gray, cmap='gray')
        plt.title('Original')
        plt.xticks([])
        plt.yticks([])

        plt.subplot(2, 3, 2)
        plt.imshow(mask, cmap='gray')
        plt.title('Mask')
        plt.xticks([])
        plt.yticks([])

        plt.subplot(2, 3, 3)
        plt.imshow(mask_inv, cmap='gray')
        plt.title('Mask_inv')
        plt.xticks([])
        plt.yticks([])

        plt.subplot(2, 3, 4)
        plt.imshow(img_masked, cmap='gray')
        plt.title('Img_masked')
        plt.xticks([])
        plt.yticks([])

        plt.subplot(2, 3, 5)
        plt.imshow(median_masked, cmap='gray')
        plt.title('Median_masked')
        plt.xticks([])
        plt.yticks([])

        plt.subplot(2, 3, 6)
        plt.imshow(result, cmap='gray')
        plt.title('Result')
        plt.xticks([])
        plt.yticks([])

        plt.show()

    return result


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


# вспомогательные флаги-переменные
flag_convert = False
flag_filter = False
image_uploaded = False
field_uploaded = False

coef_scale = None  # коэффициент масштабирования метр/пиксель
"""
while not image_uploaded:
    # получение имени файла и пути к нему
    if Test:
        image_name = tests[nametest]['image']
    else:
        Tk().withdraw()
        image_name = askopenfilename()
    path_with_name, file_extension = os.path.splitext(image_name)
    folder, filename = os.path.split(path_with_name)

    # чтение изображения из файла
    try:
        # читаем изображение в оттенках серого
        image_file = Image.open(image_name).convert('L')
        img_original = np.asarray(image_file)
        # сохраняем изображение как массив
        imgorig = np.asarray(image_file)

        image_file.show()
        # если изображение не формата .jpeg, необходимо его конвертировать
        if file_extension != '.jpg' and file_extension != '.JPG' \
                and file_extension != '.jpeg' and file_extension != '.JPEG':
            # читаем файл
            imgorig = cv.imread(image_name, cv.IMREAD_ANYCOLOR | cv.IMREAD_ANYDEPTH)
            #cv.imshow('show', imgForConvert)
            #  временное сохранение изображения в формате .jpeg в рабочей папке
            #cv.imwrite('modified_img.jpg', imgForConvert, [int(cv.IMWRITE_JPEG_QUALITY), 100])
            #image_name = 'modified_img.jpg'
            #del imgForConvert
            #flag_png = True
            # exifScale = False
            # if file_extension == '.tif' or file_extension == '.TIF':
            #     exifScale = True
            #     exif_data = Image.open(image_name).getexif()
            #     try:
            #         tags = exif_data.items()._mapping[34118]
            #
            #         width = 0.
            #         measurement_dict = {
            #             "m": 1E-3,
            #             "µ": 1E-6,
            #             "n": 1E-9,
            #         }
            #
            #         for row in tags.split("\r\n"):
            #             key_value = row.split(" = ")
            #             if len(key_value) == 2:
            #                 if key_value[0] == "Width":
            #                     width = key_value[1]
            #                     break
            #         width_str = width.split(" ")
            #
            #         width_unit = width_str[1]
            #         width_value = float(width_str[0]) * measurement_dict[width_unit[0]]
            #         coef_scale = width_value / float(exif_data.items()._mapping[256])
            #     except Exception:
            #         exifScale = False
            # if not exifScale:
            #     n = 0
            #     while img_original[700, 21 + n + 1] > 240:
            #         n += 1
            #     n += 5
            #     print('Enter the size of the scale segment in micrometer')
            #     size_segment = float(input()) * 1e-6
            #     coef_scale = size_segment / n

                # Я ОСТАНОВИЛСЯ ЗДЕСЬ

            img = np.asarray(image_file.crop((0, 0, image_file.size[0], image_file.size[1] - 103)))
            imgorig = np.delete(imgorig, imgorig[-103:], axis=0)
        else:
        # чтение изображение
            imgorig = cv.imread(image_name)
            img = cv.imread(image_name)
        #cv.imshow('res', imgorig)
    except Exception as e:
        # вызывается исключение, если на вход был подан не поддерживаемый формат изображения
        mb.showerror("Error", 'Invalid image file')
        continue

    # если была совершена конвертация, временный файл с изображением в формате .jpeg удаляется
    # if flag_convert:
    #    os.remove('modified_img.jpg')

    image_uploaded = True

    if Test:
        tfi = tests[nametest]['filtered']
    else:
        print('Clear the image from noise? 0 - Yes, 1 - No\n')
        tfi = int(input())
        # 1 - Шумы на изображении будут обработаны
        # 0 - Изображение останется с шумами
    if not tfi:
        # вывод гистограммы распределения интенсивности яркостей пикселей в изображении,
        # по которой можно определить пороговые значения интенсивности
        plt.hist(img.ravel(), 256, [0, 256])
        plt.show()

        # значения по умолчанию
        up = 170
        low = 30

        print('Default parameters for noise (upper=170, lower=30)? 1 - Yes, 0 - No')
        if int(input()) == 0:
            print('Enter upper limit - ', end='')
            up = int(input())
            print('Enter lower limit - ', end='')
            low = int(input())

        # обработка изображения (удаление шумов)
        img = noise(img, lower=low, upper=up, plot=True)

        # перевод изображения в пространство оттенков серого
        # img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
        flag_filter = True

        print('Save filtered image? 1 - Yes, 0 - No\n')
        # 1 - Обработанное изображение будет сохранено
        sfi = int(input())
        if sfi == 1:
            # сохранение обработанного изображение

            filter_result_file = image_file.copy()
            filter_result_file.paste(Image.fromarray(img))
            filter_result_file.save(folder + '/ [Filtered] ' + filename + '.jpg')
    # else:
    # перевод изображения в пространство оттенков серого
    # img = cv.cvtColor(imgorig, cv.COLOR_BGR2GRAY)
    # img = imgorig
"""

while not image_uploaded:
    # получение имени файла и пути к нему
    Tk().withdraw()
    image_name = askopenfilename()
    path_with_name, file_extension = os.path.splitext(image_name)
    folder, filename = os.path.split(path_with_name)

    # чтение изображения из файла
    try:
        # если изображение не формата .jpeg, необходимо его конвертировать
        if file_extension != '.jpg' and file_extension != '.JPG' \
                and file_extension != '.jpeg' and file_extension != '.JPEG':
            # читаем файл
            imgForConvert = cv.imread(image_name)
            # временное сохранение изображения в формате .jpeg в рабочей папке
            cv.imwrite('modified_img.jpg', imgForConvert, [int(cv.IMWRITE_JPEG_QUALITY), 100])
            image_name = 'modified_img.jpg'
            del imgForConvert
            flag_png = True

        # чтение изображение
        imgorig = cv.imread(image_name)
    except Exception as e:
        # вызывается исключение, если на вход был подан не поддерживаемый формат изображения
        mb.showerror("Error", 'Invalid image file')
        continue

    # если была совершена конвертация, временный файл с изображением в формате .jpeg удаляется
    if flag_convert:
        os.remove('modified_img.jpg')

    image_uploaded = True

    print('Clear the image from noise? 1 - Yes, 0 - No\n')
    tfi = int(input())
    # 1 - Шумы на изображении будут обработаны
    # 0 - Изображение останется с шумами
    if tfi == 1:
        # вывод гистограммы распределения интенсивности яркостей пикселей в изображении,
        # по которой можно определить пороговые значения интенсивности
        plt.hist(imgorig.ravel(), 256, [0, 256])
        plt.show()

        # значения по умолчанию
        up = 170
        low = 30

        print('Default parameters for noise (upper=170, lower=30)? 1 - Yes, 0 - No')
        if int(input()) == 0:
            print('Enter upper limit - ', end='')
            up = int(input())
            print('Enter lower limit - ', end='')
            low = int(input())

        # обработка изображения (удаление шумов)
        img = noise(imgorig, lower=low, upper=up, plot=True)

        # перевод изображения в пространство оттенков серого
        img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
        flag_filter = True
    else:
        # перевод изображения в пространство оттенков серого
        img = cv.cvtColor(imgorig, cv.COLOR_BGR2GRAY)


# вывод загруженного/обработанного изображения
plt.imshow(img, cmap='gray')

if flag_filter:
    plt.title("Input filtered image")
else:
    plt.title('Input image')
plt.show()

# создаём объект для фрактального анализа
fractal = FractalAnalysis(show_step=True)

if not Test:
    print("Upload fractal field? 1 - Yes, 0 - No")
    comm = int(input())
    # 1 - Прочитать поле размерностей из файла и ввести его в анализатор
    # 0 - Поле будет создано анализатором
else:
    comm = 1

if comm == 1:
    while not field_uploaded:

        if Test:
            uploadfilename = tests[nametest]['field']
            win_size = tests[nametest]['win_size']
            dx = tests[nametest]['dx']
            dy = tests[nametest]['dy']
        else:
            # чтение фрактального поля из файла
            Tk().withdraw()
            uploadfilename = askopenfilename()
            print('Enter size of window for uploaded field - ', end='')
            win_size = int(input())
            print('Enter dx for uploaded field - ', end='')
            dx = int(input())
            print('Enter dy for uploaded field - ', end='')
            dy = int(input())

        field = np.loadtxt(uploadfilename, delimiter=";")

        # визуализация поля
        field_to_image(field, show=True)

        try:
            # загрузка в анализатор поля и его параметров
            fractal.set_field(img, win_size, dx, dy, field=field)
        except ValueError:
            # вызывается исключение, если размер поля не соответствует размерам изображения
            mb.showerror("Error", 'The shape of the field does not match the shape of the image.\nTry again!')
            continue
        field_uploaded = True
else:
    # ввод параметров поля
    print('Enter size of window - ', end='')
    win_size = int(input())
    print('Enter dx - ', end='')
    dx = int(input())
    print('Enter dy - ', end='')
    dy = int(input())

    # создание поля
    fractal.set_field(img, win_size, dx, dy, method='prism')

    # визуализация полученного поля
    field_to_image(fractal.field, show=True)

    print('Save resulting field? 1 - Yes, 0 - No')
    # 1 - Сохранить полученное поле в файл .csv
    sf = int(input())
    if sf == 1:
        print("Add datafile's name\n")
        datafilename = input()
        np.savetxt(folder + '/' + datafilename + '.csv', fractal.field, delimiter=";", newline="\n")

# вывод гистограммы распределений фрактальных размерностей в поле
ax = plt.subplot(1, 1, 1)
ax.hist(fractal.field.flatten(), bins=250, color='yellow', edgecolor='black')
ax.set_title('Distribution of fractal dimensions')
plt.tight_layout()
plt.show()


print('EM analysis? 1 - Yes, 0 - No\n')
eman = int(input())
if eman:
    em_analysis(fractal.field)


print('Exit - 0, or\nEnter number components in a mixture of distributions - ', end='')
n_comp = int(input())

if n_comp != 0:
    # сегментация переходного слоя на изображении
    # n_comp=2 - сегментация по сильному изменению поля
    # n_comp=3 - сегментация по EM-классификации
    mask_stratum = fractal.segment_stratum(n_comp)

    plt.imshow(cv.cvtColor(mask_stratum, cv.COLOR_GRAY2RGB))
    plt.show()

    img = cv.cvtColor(img, cv.COLOR_BGR2RGB)
    # создание слоя-заливки
    color_layer = np.zeros(img.shape, dtype=np.uint8)
    color_layer[:] = (30, 21, 117) if n_comp == 2 else (255, 0, 102)

    # наложение заливки на изображение через маску
    # получаются закрашенные сегменты переходного слоя
    stratum = cv.bitwise_and(img, color_layer, mask=mask_stratum)

    # соединение исходного изображения с полупрозрачными закрашенными сегментами
    result = cv.addWeighted(img, 1, stratum, 0.2, 0.0)

    # выделение контуров сегментов
    color_contours = (0, 255, 255) if n_comp == 2 else (102, 0, 153)
    contours, hierarchy = cv.findContours(mask_stratum, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
    cv.drawContours(result, contours, -1, color_contours, 1, cv.LINE_AA, hierarchy, 3)

    # перевод результата в пространство RGB
    result = cv.cvtColor(result, cv.COLOR_BGR2RGB)

    color_square = (1, 1, 0) if n_comp == 2 else (0.4, 0, 0.6)
    plt.figure(figsize=(13, 7))
    plt.imshow(result)
    purple_square = lns.Line2D([], [], color=color_square, marker='s', linestyle='None',
                               markersize=7, label='Переходный слой')
    plt.legend(handles=[purple_square])

    """
    #if n_comp == 2:
        #distances, coor = fractal.distances_curves
        #distances2, coor2 = fractal.distances_curves_vertical
        #mean_d_1 = np.mean(distances)
        #mean_d_2 = np.mean(distances2)
        #print("Средняя ширина переходного слоя по первому методу: {:e}".format(mean_d_1 * coef_scale))
        #print("Средняя ширина переходного слоя по второму методу: {:e}".format(mean_d_2 * coef_scale))
        
        for i, d in enumerate(distances):
            plt.text(coor[i, 0, 0], coor[i, 0, 1], "{:e}".format(d * coef_scale),
                     bbox=dict(boxstyle="round,pad=0.3", fc="#d69f67", ec="black", lw=1))
        for i, d in enumerate(distances2):
            plt.text(coor2[i, 1, 0], coor2[i, 1, 1], "{:e}".format(d * coef_scale),
                     bbox=dict(boxstyle="round,pad=0.3", fc="#674D92", ec="white", lw=1))
        """
    plt.show()

