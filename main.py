from tkinter import Tk
from tkinter.filedialog import askopenfilename, askdirectory
import os.path
import cv2 as cv
import tkinter.messagebox as mb
import matplotlib.pyplot as plt
import numpy as np
from FractalAnalysisClass import FractalAnalysis


def noice(img, lower, upper, plot=False):
    """
    Функция избавляется от шумов на изображении (чёрные или белые пятна)

    :param img: 3-D numpy массив.
        Изображение в цветовом пространстве BGR.
    :param lower: int
        Нижний предел допустимой (не шумовой) интенсивности цвета.
    :param upper: int
        Верхний предел допустимой (не шумовой) интенсивности цвета.
    :param plot: bool.
        True, если необходимо вывести пошаговую обработку изображения.
        False, если это не нужно.
                    
    :return: 3-D numpy массив размерности NxMx3.
        Обработанное изображение в цветовом пространстве BGR.
    """
    # перевод изображения в пространство оттенков серого
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

    # пороговая бинаризация изображение
    # пиксели, интенсивность которых не входит в диапазон [lower, upper], становятся чёрными. Остальные - белыми
    mask1 = cv.threshold(gray, lower, 255, cv.THRESH_BINARY_INV)[1]
    mask2 = cv.threshold(gray, upper, 255, cv.THRESH_BINARY)[1]
    mask = cv.add(mask1, mask2)
    mask = cv.bitwise_not(mask)

    # Увеличиваем радиус каждого чёрного объекта
    kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE, (5, 5))
    mask = cv.morphologyEx(mask, cv.MORPH_ERODE, kernel)

    # Делаем маску трёх-канальной
    mask = cv.merge([mask, mask, mask])

    # Инвертируем маску
    mask_inv = 255 - mask

    # Считаем радиус для размытия областей с шумом
    radius = int(img.shape[0] / 3) + 1
    if radius % 2 == 0:
        radius = radius + 1

    # Размываем изображение
    median = cv.medianBlur(img, radius)

    # Накладываем маску на изображение
    img_masked = cv.bitwise_and(img, mask)

    # Накладываем инвертированную маску на размытое изображение
    median_masked = cv.bitwise_and(median, mask_inv)

    # Соединяем два полученных изображения
    result = cv.add(img_masked, median_masked)

    if plot:
        plt.subplot(2, 3, 1)
        plt.imshow(gray, cmap='gray')
        plt.title('Original')
        plt.xticks([])
        plt.yticks([])

        plt.subplot(2, 3, 2)
        plt.imshow(mask)
        plt.title('Mask')
        plt.xticks([])
        plt.yticks([])

        plt.subplot(2, 3, 3)
        plt.imshow(mask_inv)
        plt.title('Mask_inv')
        plt.xticks([])
        plt.yticks([])

        plt.subplot(2, 3, 4)
        plt.imshow(img_masked)
        plt.title('Img_masked')
        plt.xticks([])
        plt.yticks([])

        plt.subplot(2, 3, 5)
        plt.imshow(median_masked)
        plt.title('Median_masked')
        plt.xticks([])
        plt.yticks([])

        plt.subplot(2, 3, 6)
        plt.imshow(result)
        plt.title('Result')
        plt.xticks([])
        plt.yticks([])

        plt.show()

    return result

def field_to_image(input_field, show=False):
    """
    Перевод поля фрактальных размерностей в пространство оттенков серого
    
    :param input_field:     2-D numpy массив
        Поле фрактальных размерностей
    :param show:            bool
        True, если необходимо вывести изображение визуализации поля
    
    :return:                2-D numpy массив
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

# если была совершена конвертация, временный файл с изображением в формате .jpeg удаляется
if flag_convert:
    os.remove('modified_img.jpg')

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
    
    print('Default parameters for noice (upper=170, lower=30)? 1 - Yes, 0 - No')
    if int(input()) == 0:
        print('Enter upper limit - ', end='')
        up = int(input())
        print('Enter lower limit - ', end='')
        low = int(input())
        
    # обработка изображения (удаление шумов)
    img = noice(imgorig, lower=low, upper=up, plot=True)
    
    # перевод изображения в пространство оттенков серого
    img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    flag_filter = True
else:
    # перевод изображения в пространство оттенков серого
    img = cv.cvtColor(imgorig, cv.COLOR_BGR2GRAY)

del imgorig

# вывод загруженного/обработанного изображения
plt.imshow(img, cmap='gray')
if flag_filter:
    plt.title("Input filtered image")
else:
    plt.title('Input image')
plt.show()

if flag_filter:
    print('Save filtered image? 1 - Yes, 0 - No\n')
    # 1 - Обработанное изображение будет сохранено
    sfi = int(input())
    if sfi == 1:
        # сохранение обработанного изображение
        cv.imwrite(folder + '/ [Filtered] ' + filename + '.jpg', img)

# создаём объект для фрактального анализа
fractal = FractalAnalysis(show_step=True)

print("Upload fractal field? 1 - Yes, 0 - No")
comm = int(input())
# 1 - Прочитать поле размерностей из файла и ввести его в анализатор
# 0 - Поле будет создано анализатором
if comm == 1:
    # чтение фрактального поля из файла
    Tk().withdraw()
    uploadfilename = askopenfilename()
    field = np.loadtxt(uploadfilename, delimiter=";")

    print('Enter size of window for uploaded field - ', end='')
    win_size = int(input())
    print('Enter dx for uploaded field - ', end='')
    dx = int(input())
    print('Enter dy for uploaded field - ', end='')
    dy = int(input())

    # визуализация поля
    field_to_image(field, show=True)

    # загрузка в анализатор поля и его параметров
    fractal.set_field(img, win_size, dx, dy, field=field)
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

# ввод количества распределений в смеси
print('Enter number components in a mixture of distributions - ', end='')
n_comp = int(input())

# сегментация переходного слоя на изображении
# n_comp=2 - сегментация по сильному изменению поля
# n_comp=3 - сегментация по EM-классификации
fractal.segment_stratum(n_comp, folder=folder)

"""
# get area of largest contour
    contours = cv.findContours(mask_inv[:, :, 0], cv.RETR_EXTERNAL, cv.CHAIN_APPROX_NONE)
    contours = contours[0] if len(contours) == 2 else contours[1]
    perimeter_max = 0
    for c in contours:
        perimeter = cv.arcLength(c, True)
        if perimeter > perimeter_max:
            perimeter_max = perimeter
            
            
import csv
import os
from math import ceil, sqrt, log
from tkinter import Tk
from tkinter.filedialog import askopenfilename, askdirectory
import statsmodels.api as sm

import cv2 as cv
import matplotlib.pyplot as plt
import matplotlib.lines as lns
import numpy
import numpy as np
<<<<<<< HEAD
<<<<<<< HEAD
from scipy import signal
from sklearn.mixture import GaussianMixture as GMM
from sklearn.cluster import DBSCAN
from sklearn.neighbors import NearestNeighbors
from kneed import KneeLocator
from time import strftime

from FractalAnalysisClass import FractalAnalysis



# create of windows
=======
=======
>>>>>>> 4e9401b4c0aba1445f05a0b4f173a62ade31b2c9
from math import ceil, sqrt, log
import matplotlib.pyplot as plt
import cv2 as cv
from tkinter import Tk
from tkinter.filedialog import askopenfilename, askdirectory
from scipy import signal


# creater of windows
<<<<<<< HEAD
>>>>>>> 4e9401b4c0aba1445f05a0b4f173a62ade31b2c9
=======
>>>>>>> 4e9401b4c0aba1445f05a0b4f173a62ade31b2c9
def roll(a,  # ND array
         b,  # rolling 2D window array
         dx,  # horizontal step, abscissa, number of columns
         dy):  # vertical step, ordinate, number of rows
    shape = a.shape[:-2] + \
            ((a.shape[-2] - b.shape[-2]) // dy + 1,) + \
            ((a.shape[-1] - b.shape[-1]) // dx + 1,) + \
            b.shape  # sausage-like shape with 2D cross-section
    strides = a.strides[:-2] + \
              (a.strides[-2] * dy,) + \
              (a.strides[-1] * dx,) + \
              a.strides[-2:]
    return np.lib.stride_tricks.as_strided(a, shape=shape, strides=strides)


<<<<<<< HEAD
def _method_cubes(windows, e):
    # for row in windows:
    #    for t_win in row:
=======
def method_cubes(windows, e):
    print('count = ', windows.shape[0], ' : ', windows.shape[1])
>>>>>>> 4e9401b4c0aba1445f05a0b4f173a62ade31b2c9
    for i in range(windows.shape[0]):
        for j in range(windows.shape[1]):
            t_win = windows[i][j]
            n = np.zeros(e.shape)
            for ei in range(e.shape[0]):
                size = e[ei]
                n_e = 0
                n_x = ceil(t_win.shape[1] / size)
                n_y = ceil(t_win.shape[0] / size)
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
                        srez = t_win[y_beg:y_end, x_beg:x_end]
                        max = np.amax(srez)
                        min = np.amin(srez)
                        if max != min:
                            n_e += ceil(max / size) - ceil(min / size) + 1
                        else:
                            n_e += 1
                n[ei] = n_e
            lgn = np.zeros(n.shape)
            lge = np.zeros(e.shape)
<<<<<<< HEAD
<<<<<<< HEAD
=======
            # lgre = np.zeros(e.shape)
>>>>>>> 4e9401b4c0aba1445f05a0b4f173a62ade31b2c9
=======
            # lgre = np.zeros(e.shape)
>>>>>>> 4e9401b4c0aba1445f05a0b4f173a62ade31b2c9
            for k in range(e.shape[0]):
                lgn[k] = log(n[k])
                lge[k] = log(e[k])
            A = np.vstack([lge, np.ones(len(lge))]).T
            D, c = np.linalg.lstsq(A, lgn, rcond=None)[0]
            print(-D)
            pole[i][j] = -D
        print('\rMethod of cubes [', '█' * int(i / windows.shape[0] * 20),
              ' ' * int((windows.shape[0] - i) / windows.shape[0] * 20), ']\t', i, '\t/ ', windows.shape[0] - 1, end='')
        print('\n')
    return pole


<<<<<<< HEAD
<<<<<<< HEAD
def _method_prism(windows, s):
    field = np.zeros((windows.shape[0], windows.shape[1]))
    for i in range(windows.shape[0]):
        for j in range(windows.shape[1]):
            t_win = windows[i][j]
=======
=======
>>>>>>> 4e9401b4c0aba1445f05a0b4f173a62ade31b2c9
def method_prism(windows, s):
    print('count = ', windows.shape[0], ' : ', windows.shape[1])
    for i in range(windows.shape[0]):
        for j in range(windows.shape[1]):
            t_win = windows[i][j]
            print('window ', i, ' : ', j)
<<<<<<< HEAD
>>>>>>> 4e9401b4c0aba1445f05a0b4f173a62ade31b2c9
=======
>>>>>>> 4e9401b4c0aba1445f05a0b4f173a62ade31b2c9
            Se = np.zeros(s.shape)
            for si in range(s.shape[0]):
                size = s[si]
                S = 0
                n_x = ceil(t_win.shape[1] / size)
                n_y = ceil(t_win.shape[0] / size)
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
                        srez = t_win[y_beg:y_end, x_beg:x_end]

                        sx = srez.shape[1]
                        sy = srez.shape[0]
                        sdi = sqrt(sx ** 2 + sy ** 2)

                        a = int(srez[0][0])
                        b = int(srez[0][sx - 1])
                        c = int(srez[sy - 1][0])
                        d = int(srez[sy - 1][sx - 1])
                        e = int(srez[int((sy - 1) / 2)][int((sx - 1) / 2)])
                        e = (a + b + c + d) / 4

                        ae = sqrt((a - e) ** 2 + (sdi / 2) ** 2)
                        be = sqrt((b - e) ** 2 + (sdi / 2) ** 2)
                        ce = sqrt((c - e) ** 2 + (sdi / 2) ** 2)
                        de = sqrt((d - e) ** 2 + (sdi / 2) ** 2)

                        ab = sqrt((b - a) ** 2 + sx ** 2)
                        bd = sqrt((b - d) ** 2 + sy ** 2)
                        cd = sqrt((c - d) ** 2 + sx ** 2)
                        ac = sqrt((a - c) ** 2 + sy ** 2)

                        pA = (ab + be + ae) / 2
                        pB = (bd + be + de) / 2
                        pC = (cd + de + ce) / 2
                        pD = (ac + ce + ae) / 2

                        SA = sqrt(pA * (pA - ab) * (pA - be) * (pA - ae))
                        SB = sqrt(pB * (pB - bd) * (pB - be) * (pB - de))
                        SC = sqrt(pC * (pC - cd) * (pC - de) * (pC - ce))
                        SD = sqrt(pD * (pD - ac) * (pD - ce) * (pD - ae))

                        S += SA + SB + SC + SD
                Se[si] = S

<<<<<<< HEAD
<<<<<<< HEAD
=======
=======
>>>>>>> 4e9401b4c0aba1445f05a0b4f173a62ade31b2c9
            e_sr = 0
            for sii in s:
                e_sr += sii
            e_sr = e_sr / (s.shape[0])

            S_sr = 0
            for Sei in Se:
                S_sr += Sei
            S_sr = (S_sr / Se.shape[0])

            cov = 0
            for sii, Si in zip(s, Se):
                cov += (sii - e_sr) * (Si - S_sr)
            cov /= (Se.shape[0] - 1)

            S_e = 0
            for sii in s:
                S_e += (sii - e_sr) ** 2
            S_e /= (s.shape[0] - 1)
            S_e = sqrt(S_e)

            S_S = float(0.0)
            for Sei in Se:
                S_S += (Sei - S_sr) ** 2
            S_S /= (Se.shape[0] - 1)
            S_S = sqrt(S_S)

            r = cov / S_e / S_S

            B = r * S_S / S_e

            D1 = 2 - B

<<<<<<< HEAD
>>>>>>> 4e9401b4c0aba1445f05a0b4f173a62ade31b2c9
=======
>>>>>>> 4e9401b4c0aba1445f05a0b4f173a62ade31b2c9
            lgS = np.zeros(Se.shape)
            lge = np.zeros(s.shape)
            for k in range(s.shape[0]):
                lgS[k] = log(Se[k])
                lge[k] = log(s[k])
<<<<<<< HEAD
<<<<<<< HEAD
            A = np.vstack([lge, np.ones(len(lge))]).T
            D, Cc = np.linalg.lstsq(A, lgS, rcond=None)[0]
            D2 = 2 - D
            field[i][j] = D2
        print('\rMethod of prism [', '█' * int(i / windows.shape[0] * 20),
              ' ' * int((windows.shape[0] - i) / windows.shape[0] * 20), ']\t', i, '\t/ ', windows.shape[0] - 1, end='')
    print('\n')
    return field



=======
            # plt.plot(lge, lgS)
            # plt.show()
            A = np.vstack([lge, np.ones(len(lge))]).T
            D, Cc = np.linalg.lstsq(A, lgS, rcond=None)[0]
            D2 = 2 - D
            print(D1)
            print(D2)
            pole[i][j] = D2
    return pole


=======
            # plt.plot(lge, lgS)
            # plt.show()
            A = np.vstack([lge, np.ones(len(lge))]).T
            D, Cc = np.linalg.lstsq(A, lgS, rcond=None)[0]
            D2 = 2 - D
            print(D1)
            print(D2)
            pole[i][j] = D2
    return pole


>>>>>>> 4e9401b4c0aba1445f05a0b4f173a62ade31b2c9
def showpole(pole):
    imgout = np.empty((pole.shape[0], pole.shape[1]), dtype=numpy.uint8)
    for i in range(imgout.shape[0]):
        for j in range(imgout.shape[1]):

            if pole[i][j] - 1.0 > 0.0:
                imgout[i][j] = int(255 * (pole[i][j] - 1) / 3)
            else:
                imgout[i][j] = 0
    matplotlib.pyplot.imshow(imgout, cmap='gray')
    plt.show()


def goteststocsv():
    name = 'test 4'
    win_size = 20
    dx = 3
    dy = 3

    imgorig = matplotlib.pyplot.imread("C:/Users/bortnikov.2018/Desktop/JPEG/Контрольные тесты/" + name + ".jpg")
    A = np.delete(imgorig, (1, 2), 2)
    A = np.resize(A, (imgorig.shape[0], imgorig.shape[1]))

    b = np.zeros((win_size, win_size))
    windows = roll(A, b, dx, dy)  # array of windows
    e = np.array([win_size, int((win_size + 1) / 2), int((win_size + 3) / 4)])

    cv.imshow(name, windows[0][0])
    cv.waitKey()
    cv.destroyAllWindows()

    pole20 = method_cubes(windows, e)
    plt.hist(pole20)
    plt.show()

    showpole(pole20)

    p20 = pole20.flatten()

    win_size = 10
    b = np.zeros((win_size, win_size))
    windows = roll(A, b, dx, dy)  # array of windows
    e = np.array([win_size, int((win_size + 1) / 2), int((win_size + 3) / 4)])

    cv.imshow(name, windows[0][0])
    cv.waitKey()
    cv.destroyAllWindows()

    pole10 = method_cubes(windows, e)
    plt.hist(pole10)
    plt.show()

    showpole(pole10)

    p10 = pole10.flatten()

    win_size = 30
    b = np.zeros((win_size, win_size))
    windows = roll(A, b, dx, dy)  # array of windows
    e = np.array([win_size, int((win_size + 1) / 2), int((win_size + 3) / 4)])

    cv.imshow(name, windows[0][0])
    cv.waitKey()
    cv.destroyAllWindows()

    pole30 = method_cubes(windows, e)
    plt.hist(pole30)
    plt.show()
    showpole(pole30)

    p30 = pole30.flatten()

    win_size = 40
    b = np.zeros((win_size, win_size))
    windows = roll(A, b, dx, dy)  # array of windows
    e = np.array([win_size, int((win_size + 1) / 2), int((win_size + 3) / 4)])

    cv.imshow(name, windows[0][0])
    cv.waitKey()
    cv.destroyAllWindows()

    pole40 = method_cubes(windows, e)
    plt.hist(pole40)
    plt.show()
    showpole(pole40)

    p40 = pole40.flatten()

    win_size = 50
    b = np.zeros((win_size, win_size))
    windows = roll(A, b, dx, dy)  # array of windows
    e = np.array([win_size, int((win_size + 1) / 2), int((win_size + 3) / 4)])

    cv.imshow(name, windows[0][0])
    cv.waitKey()
    cv.destroyAllWindows()

    pole50 = method_cubes(windows, e)
    plt.hist(pole50)
    plt.show()
    showpole(pole50)

    p50 = pole50.flatten()

    win_size = 60
    b = np.zeros((win_size, win_size))
    windows = roll(A, b, dx, dy)  # array of windows
    e = np.array([win_size, int((win_size + 1) / 2), int((win_size + 3) / 4)])

    cv.imshow(name, windows[0][0])
    cv.waitKey()
    cv.destroyAllWindows()

    pole60 = method_cubes(windows, e)
    plt.hist(pole60)
    plt.show()
    showpole(pole60)

    p60 = pole60.flatten()

    with open("C:/Users/bortnikov.2018/Desktop/JPEG/" + name + ".csv", mode="w", encoding='utf-8') as w_file:
        file_writer = csv.writer(w_file, delimiter=';', lineterminator='\n')
        file_writer.writerow(('10', '20', '30', '40', '50', '60'))
        for i in range(p60.shape[0]):
            file_writer.writerow((p10[i], p20[i], p30[i], p40[i], p50[i], p60[i]))
        for i in range(p60.shape[0], p50.shape[0]):
            file_writer.writerow((p10[i], p20[i], p30[i], p40[i], p50[i]))
        for i in range(p50.shape[0], p40.shape[0]):
            file_writer.writerow((p10[i], p20[i], p30[i], p40[i]))
        for i in range(p40.shape[0], p30.shape[0]):
            file_writer.writerow((p10[i], p20[i], p30[i]))
        for i in range(p30.shape[0], p20.shape[0]):
            file_writer.writerow((p10[i], p20[i]))
        for i in range(p20.shape[0], p10.shape[0]):
            file_writer.writerow((p10[i],))


def goteststocsv_prism(name):
    win_size = 20
    dx = 3
    dy = 3

    imgorig = matplotlib.pyplot.imread("C:/Users/bortnikov.2018/Desktop/JPEG/Контрольные тесты/" + name + ".jpg")
    A = np.delete(imgorig, (1, 2), 2)
    A = np.resize(A, (imgorig.shape[0], imgorig.shape[1]))
    '''
    Af = A.flatten()
    with open("C:/Users/bortnikov.2018/Desktop/JPEG/" + name + "_px.csv", mode="w", encoding='utf-8') as w_file:
        file_writer = csv.writer(w_file, delimiter=';', lineterminator='\n')
        for i in range(Af.shape[0]):
            file_writer.writerow((Af[i],))
    '''
    '''
    b = np.zeros((win_size, win_size))
    windows = roll(A, b, dx, dy)  # array of windows
    print(windows.shape)
    e = np.array([win_size, int((win_size + 1) / 2), int((win_size + 3) / 4)])


    cv.imshow(name, windows[0][0])
    cv.waitKey()
    cv.destroyAllWindows()

    pole20 = method_prism(windows, e)
    plt.hist(pole20)
    plt.show()
    showpole(windows, pole20)

    p20 = pole20.flatten()

    win_size = 10
    b = np.zeros((win_size, win_size))
    windows = roll(A, b, dx, dy)  # array of windows
    print(windows.shape)
    e = np.array([win_size, int((win_size + 1) / 2), int((win_size + 3) / 4)])

    cv.imshow(name, windows[0][0])
    cv.waitKey()
    cv.destroyAllWindows()

    pole10 = method_prism(windows, e)
    plt.hist(pole10)
    plt.show()
    showpole(windows, pole10)

    p10 = pole10.flatten()


    win_size = 30
    b = np.zeros((win_size, win_size))
    windows = roll(A, b, dx, dy)  # array of windows
    print(windows.shape)
    e = np.array([win_size, int((win_size + 1) / 2), int((win_size + 3) / 4)])

    cv.imshow(name, windows[0][0])
    cv.waitKey()
    cv.destroyAllWindows()

    pole30 = method_prism(windows, e)
    plt.hist(pole30)
    plt.show()
    showpole(windows, pole30)
    p30 = pole30.flatten()

    win_size = 40
    b = np.zeros((win_size, win_size))
    windows = roll(A, b, dx, dy)  # array of windows
    e = np.array([win_size, int((win_size + 1) / 2), int((win_size + 3) / 4)])
    print(windows.shape)


    cv.imshow(name, windows[0][0])
    cv.waitKey()
    cv.destroyAllWindows()

    pole40 = method_prism(windows, e)
    plt.hist(pole40)
    plt.show()
    showpole(windows, pole40)

    p40 = pole40.flatten()


    win_size = 50
    b = np.zeros((win_size, win_size))
    windows = roll(A, b, dx, dy)  # array of windows
    e = np.array([win_size, int((win_size + 1) / 2), int((win_size + 3) / 4)])
    print(windows.shape)


    cv.imshow(name, windows[0][0])
    cv.waitKey()
    cv.destroyAllWindows()

    pole50 = method_prism(windows, e)
    plt.hist(pole50)
    plt.show()
    showpole(windows, pole50)
    p50 = pole50.flatten()

    win_size = 60
    b = np.zeros((win_size, win_size))
    windows = roll(A, b, dx, dy)  # array of windows
    e = np.array([win_size, int((win_size + 1) / 2), int((win_size + 3) / 4)])
    print(windows.shape)


    cv.imshow(name, windows[0][0])
    cv.waitKey()
    cv.destroyAllWindows()

    pole60 = method_prism(windows, e)
    plt.hist(pole60)
    plt.show()
    showpole(windows, pole60)
    p60 = pole60.flatten()
    '''

    win_size = 30
    b = np.zeros((win_size, win_size))
    windows = roll(A, b, dx, dy)  # array of windows
    e = np.array([win_size, int((win_size + 1) / 2), int((win_size + 3) / 4)])
    print(windows.shape)

    cv.imshow(name, windows[0][0])
    cv.waitKey()
    cv.destroyAllWindows()

    pole70 = method_prism(windows, e)
    plt.hist(pole70)
    plt.show()
    showpole(pole70)
    p70 = pole70.flatten()
    '''
    with open("C:/Users/bortnikov.2018/Desktop/JPEG/" + name + ".csv", mode="w", encoding='utf-8') as w_file:
        file_writer = csv.writer(w_file, delimiter=';', lineterminator='\n')
        file_writer.writerow(('10', '20', '30', '40', '50', '60'))
        for i in range(p70.shape[0]):
            file_writer.writerow((p10[i], p20[i], p30[i], p40[i], p50[i], p60[i], p70[i]))
        for i in range(p70.shape[0], p60.shape[0]):
            file_writer.writerow((p10[i], p20[i], p30[i], p40[i], p50[i], p60[i]))
        for i in range(p60.shape[0], p50.shape[0]):
            file_writer.writerow((p10[i], p20[i], p30[i], p40[i], p50[i]))
        for i in range(p50.shape[0], p40.shape[0]):
            file_writer.writerow((p10[i], p20[i], p30[i], p40[i]))
        for i in range(p40.shape[0], p30.shape[0]):
            file_writer.writerow((p10[i], p20[i], p30[i]))
        for i in range(p30.shape[0], p20.shape[0]):
            file_writer.writerow((p10[i], p20[i]))
        for i in range(p20.shape[0], p10.shape[0]):
            file_writer.writerow((p10[i],))
    '''
<<<<<<< HEAD
>>>>>>> 4e9401b4c0aba1445f05a0b4f173a62ade31b2c9
=======
>>>>>>> 4e9401b4c0aba1445f05a0b4f173a62ade31b2c9


def find_extr(data):
    mask = np.zeros(data.size, dtype=numpy.int8)
    extr = np.empty(0, dtype=numpy.uint32)
    ampl = np.empty(0, dtype=numpy.float64)
    i = 0
    toMax = False
    while data[i + 1] == data[i] and i + 1 != data.size:
        i += 1
    extr = np.append(extr, [i])
    if i + 1 != data.size:
        if data[i + 1] > data[i]:
            mask[i] = -1
            toMax = True
        else:
            mask[i] = 1
            toMax = False
    while i + 1 != data.size:
        if toMax == True:
            while i + 1 != data.size and data[i + 1] >= data[i]:
                i += 1
            mask[i] = 1
            toMax = False
        else:
            while i + 1 != data.size and data[i + 1] <= data[i]:
                i += 1
            mask[i] = - 1
            toMax = True
        ampl = np.append(ampl, [abs(data[i] - data[extr[-1]])])
        extr = np.append(extr, [i])
<<<<<<< HEAD
<<<<<<< HEAD
    return ampl, extr


def poleToVector(pole):
    p = pole.flatten()
    Tk().withdraw()
    DIRname = askdirectory()

    with open(DIRname + "/vector.csv", mode="w", encoding='utf-8') as w_file:
        file_writer = csv.writer(w_file, delimiter=';', lineterminator='\n')
        for i in range(p.shape[0]):
            file_writer.writerow((p[i],))


def EM(X, component=2):
    model = GMM(n_components=component, covariance_type='full')
    model.fit(X)
    temp_predict_X = model.predict(X)
    means = model.means_
    return temp_predict_X, means


def vizualizateEM(arr, n, plot=False):
    imgarr = arr / n * 255
    imgarr = imgarr.astype(np.uint8)

    if plot:
        plt.imshow(imgarr, cmap='gray')
        plt.show()

    return imgarr


def segments(img, intensity, plot=False):
    value_img = cv.inRange(img, intensity, intensity + 1)

    kernel = cv.getStructuringElement(cv.MORPH_RECT, (7, 7))
    open = cv.morphologyEx(value_img, cv.MORPH_OPEN, kernel)

    kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE, (5, 5))
    close = cv.morphologyEx(open, cv.MORPH_CLOSE, kernel)

    blur = cv.GaussianBlur(close, (29, 29), 0)

    _, thresh = cv.threshold(blur, 30, 255, cv.THRESH_BINARY)

    contours, hierarchy = cv.findContours(thresh.copy(), cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)

    cv.drawContours(thresh, contours, -1, 100, 2, cv.LINE_AA, hierarchy, 3)

    nb_components, output, stats, centroids = cv.connectedComponentsWithStats(thresh, 4, cv.CV_32S)
    meanS = np.mean(stats[1:, -1])

    img2 = np.zeros(output.shape, dtype=np.uint8)

    for i in range(1, nb_components):
        if stats[i, -1] > meanS:
            img2[output == i] = 255

    if plot:
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
        plt.imshow(open, cmap='gray')
        plt.title('Open')
        plt.xticks([])
        plt.yticks([])

        plt.subplot(2, 3, 4)
        plt.imshow(close, cmap='gray')
        plt.title('Close')
        plt.xticks([])
        plt.yticks([])

        plt.subplot(2, 3, 5)
        plt.imshow(thresh, cmap='gray')
        plt.title('Blur&Thresh')
        plt.xticks([])
        plt.yticks([])

        plt.subplot(2, 3, 6)
        plt.imshow(img2, cmap='gray')
        plt.title('Sort')
        plt.xticks([])
        plt.yticks([])

        plt.show()

    return img2


change_begin = np.array([0, 0], dtype=np.uint32)
change_end = np.array([0, 0], dtype=np.uint32)

flag_convert = False
flag_filter = False

# загрузка изображения
Tk().withdraw()
imagename = askopenfilename()
path_with_name, file_extension = os.path.splitext(imagename)
folder, filename = os.path.split(path_with_name)

# конвертация файла
if file_extension != '.jpg' and file_extension != '.JPG' and file_extension != '.jpeg' and file_extension != '.JPEG':
    imgForConvert = cv.imread(imagename)
    cv.imwrite('modified_img.jpg', imgForConvert, [int(cv.IMWRITE_JPEG_QUALITY), 100])
    imagename = 'modified_img.jpg'
    del imgForConvert
    flag_png = True

imgorig = cv.imread(imagename)

if flag_convert:
    os.remove('modified_img.jpg')

print('Clear the image from noise? 1 - Yes, 0 - No\n')
tfi = int(input())
if tfi == 1:
    A = noice(img=imgorig, plot=True)
    A = cv.cvtColor(A, cv.COLOR_BGR2GRAY)
    flag_filter = True
else:
    A = cv.cvtColor(imgorig, cv.COLOR_BGR2GRAY)

del imgorig

print('Image uploaded!')
plt.imshow(A, cmap='gray')
if flag_filter:
    plt.title("Input filtered image")
else:
    plt.title('Input image')
plt.show()

if flag_filter:
    print('Save filtered image? 1 - Yes, 0 - No\n')
    sfi = int(input())
    if sfi == 1:
        cv.imwrite(folder + '/ [Filtered] ' + filename + '.jpg', A)



# инициализация поля
print("\nCreate new pole - 1\nUpload pole - 2\nSegment3 - 3\n")
comm = int(input())
if comm == 1:
    print("Enter window's size\n")
    win_size = int(input())
    print("Enter dx\n")
    dx = int(input())
    print("Enter dy\n")
=======
=======
>>>>>>> 4e9401b4c0aba1445f05a0b4f173a62ade31b2c9
    return ampl, extr, mask

def find_extr_int(data):
    mask = np.zeros(data.size, dtype=numpy.int8)
    extr = np.empty(0, dtype=numpy.uint32)
    ampl = np.empty(0, dtype=numpy.float64)
    i = 0
    toMax = False
    while data[i + 1] == data[i] and i + 1 != data.size:
        i += 1
    extr = np.append(extr, [i])
    if i + 1 != data.size:
        if data[i + 1] > data[i]:
            mask[i] = -1
            toMax = True
        else:
            mask[i] = 1
            toMax = False
    while i + 1 != data.size:
        if toMax == True:
            while i + 1 != data.size and data[i + 1] >= data[i]:
                i += 1
            mask[i] = 1
            toMax = False
        else:
            while i + 1 != data.size and data[i + 1] <= data[i]:
                i += 1
            mask[i] = - 1
            toMax = True
        ampl = np.append(ampl, [abs(int(data[i]) - int(data[extr[-1]]))])
        extr = np.append(extr, [i])
    return ampl, extr, mask


def beg_end_pick(ind_pick, amplMF, extrMF):
    points = np.empty(0, dtype=numpy.uint32)
    for i in ind_pick:
        #j = i.copy()
        #while amplMF[i] / amplMF[j] < 4 and j > 0:
        #    j -= 1
        if i > 0:
            j = i - 1
        else:
            j = 0
        points = np.append(points, [extrMF[j + 1]])
        j = i.copy()
        #while amplMF[i] / amplMF[j] < 4 and j < amplMF.shape[0] - 1:
        #    j += 1
        points = np.append(points, [extrMF[j + 1]])
    return points


def to_global(points, N, window, dn):
    new_points = np.empty(0, dtype=numpy.uint32)
    for point in points:
        temp_point = int(point * dn + window / 2)
        if temp_point > N - window / 2:
            temp_point = N - window / 2
        new_points = np.append(new_points, [temp_point])
    return new_points



# загрузка изображения
Tk().withdraw()
# imagename = askopenfilename()
imagename = "C:/Users/bortnikov.2018/Desktop/JPEG/Контрольные тесты/test 4.jpg"
imgorig = matplotlib.pyplot.imread(imagename)
A = np.delete(imgorig, (1, 2), 2)
A = np.resize(A, (imgorig.shape[0], imgorig.shape[1]))
print('Image uploaded!')

plt.imshow(A, cmap='gray')
plt.title("Original image")
plt.show()

# инициализация поля
print("Create new pole - 1\nUpload pole - 2\n")
comm = int(input())
#comm = 1
if comm == 1:
    print("Add window's size\n")
    win_size = int(input())
    print("Add dx\n")
    dx = int(input())
    print("Add dy\n")
<<<<<<< HEAD
>>>>>>> 4e9401b4c0aba1445f05a0b4f173a62ade31b2c9
=======
>>>>>>> 4e9401b4c0aba1445f05a0b4f173a62ade31b2c9
    dy = int(input())
    # dx, dy = 3, 3
    b = np.zeros((win_size, win_size))
    windows = roll(A, b, dx, dy)  # array of windows
    e = np.array([win_size, int((win_size + 1) / 2), int((win_size + 3) / 4)])
    pole = np.zeros((windows.shape[0], windows.shape[1]))

<<<<<<< HEAD
<<<<<<< HEAD
    print("Choose method: cube - 1, prism - 2\n")
    method = int(input())
    if method == 1:
        pole = _method_cubes(windows, e)
    elif method == 2:
        pole = _method_prism(windows, e)

    poleToImage(pole)

    # Tk().withdraw()
    print("Add datafile's name\n")
    datafilename = input()

    np.savetxt(folder + '/' + datafilename + '.csv', pole, delimiter=";", newline="\n")

    print('Hist? 1 - Yes, 0 -  No')
    hist = int(input())
    if hist == 1:
        # Set up the plot
        ax = plt.subplot(1, 1, 1)

        # Draw the plot
        ax.hist(pole.flatten(), bins=250, color='yellow', edgecolor='black')

        # Title and labels
        ax.set_title('fractal dimension')
        plt.tight_layout()
        plt.show()
=======
=======
>>>>>>> 4e9401b4c0aba1445f05a0b4f173a62ade31b2c9
    plt.imshow(windows[0][0], cmap='gray')
    plt.show()

    print("Choose method: cube - 1, prism - 2\n")
    method = int(input())
    if method == 1:
        pole = method_cubes(windows, e)
    elif method == 2:
        pole = method_prism(windows, e)

    showpole(pole)

    Tk().withdraw()
    DIRname = askdirectory()
    print("Add datafile's name\n")
    datafilename = input()

    np.savetxt(DIRname + "/" + datafilename + ".csv", pole, delimiter=";", newline="\n")
<<<<<<< HEAD
>>>>>>> 4e9401b4c0aba1445f05a0b4f173a62ade31b2c9
=======
>>>>>>> 4e9401b4c0aba1445f05a0b4f173a62ade31b2c9

    print("All done!")
elif comm == 2:
    Tk().withdraw()
    uploadfilename = askopenfilename()
<<<<<<< HEAD
<<<<<<< HEAD

    pole = np.loadtxt(uploadfilename, delimiter=";")
    imgpole = poleToImage(pole)

    print('Hist? 1 - Yes, 0 -  No')
    hist = int(input())
    if hist == 1:
        # Set up the plot
        ax = plt.subplot(1, 1, 1)

        # Draw the plot
        ax.hist(pole.flatten(), bins=250, color='yellow', edgecolor='black')

        # Title and labels
        ax.set_title('fractal dimension')
        plt.tight_layout()
        plt.show()

else:
    print('Error!')

Frac = FractalAnalysis(show_step=True)
Frac.set_field(A, field=pole)
Frac.segment_stratum(3, folder=folder)

comm = -1

while comm != 0:
    print(
        "Show pole - 3\nShow image - 4\nAnalysis Columns - 5\nEM - 7\nExit - 0\n")
    comm = int(input())
    if comm == 3:
        poleToImage(pole)
    elif comm == 4:
        plt.imshow(A, cmap='gray')
        plt.show()

    elif comm == 5:
        col_of_points = np.empty(0, dtype=np.uint32)
        points = np.empty(0, dtype=np.uint32)
        outcol = []

        for col in range(pole.shape[1]):
            column = pole[:, col]
            # изменение размерностей в столбце
            if col in outcol:
                plt.plot(column)
                plt.title('D in column {}'.format(col))
                plt.show()

            # изменение размерностей в столбце
            ampl, extr, mask = find_extr(column)
            if col in outcol:
                plt.plot(ampl)
                plt.title('Diff between extr in column {}'.format(col))
                plt.show()
=======
=======
>>>>>>> 4e9401b4c0aba1445f05a0b4f173a62ade31b2c9
    # uploadfilename = "C:/Users/bortnikov.2018/Desktop/JPEG/Контрольные тесты/foo.csv"

    pole = np.loadtxt(uploadfilename, delimiter=";")
    showpole(pole)
else:
    print('Error!')





comm = -1

while (comm != 0):
    print("Show pole - 3\nShow image - 4\nAnalysis - 5\nUpload analysis - 6\nCorrection - 7\nExit - 0\n")
    comm = int(input())
    if comm == 3:
        showpole(pole)
    elif comm == 4:
        plt.imshow(A, cmap='gray')
        # plt.scatter([400], [300], s=3)
        plt.show()

    elif comm == 5:
        col_of_points = np.empty(0, dtype=numpy.uint32)
        points = np.empty(0, dtype=numpy.uint32)

        for col in range(pole.shape[1]):
            # изменение размерностей в столбце
            column = pole[:, col]
            #plt.plot(column)
            #plt.show()

            # изменение размерностей в столбце
            ampl, extr, mask = find_extr(column)
            # plt.plot(ampl)
            # plt.show()
<<<<<<< HEAD
>>>>>>> 4e9401b4c0aba1445f05a0b4f173a62ade31b2c9
=======
>>>>>>> 4e9401b4c0aba1445f05a0b4f173a62ade31b2c9

            # медианная оценка выборки
            'ВНИМАНИЕ! Параметр - окно медианного фильтра'
            MedianFilter = signal.medfilt(column, kernel_size=21)
<<<<<<< HEAD
<<<<<<< HEAD
            if col in outcol:
                plt.plot(MedianFilter)
                plt.title('[Filter] D in column {}'.format(col))
                plt.show()

            # амплитуда оценки
            amplMF, extrMF, maskMF = find_extr(MedianFilter)
            if col in outcol:
                plt.plot(extrMF[1:], amplMF)
                plt.title('[Filter] Diff between extr in column {}'.format(col))
                plt.show()
=======
=======
>>>>>>> 4e9401b4c0aba1445f05a0b4f173a62ade31b2c9
            # plt.plot(MedianFilter)
            # plt.show()

            # амплитуда оценки
            amplMF, extrMF, maskMF = find_extr(MedianFilter)
            # plt.plot(extrMF[1:], amplMF)
            # plt.show()
<<<<<<< HEAD
>>>>>>> 4e9401b4c0aba1445f05a0b4f173a62ade31b2c9
=======
>>>>>>> 4e9401b4c0aba1445f05a0b4f173a62ade31b2c9

            # определение пиков амплитудов
            'ВНИМАНИЕ! Параметр - высота амплитуды перехода'
            ind_pick = np.where(amplMF > 0.4)
<<<<<<< HEAD
<<<<<<< HEAD
            if ind_pick[0].size > 0:
                for i in ind_pick[0]:
                    j = i.copy()
                    if j != 0 and j < extrMF.shape[0] - 1:
                        if 0 < extrMF[j] < pole.shape[0] - 1 and extrMF[j + 1] < pole.shape[0] - 1:
                            change_begin = np.vstack([change_begin, np.array([col, extrMF[j]])])
                            change_end = np.vstack([change_end, np.array([col, extrMF[j + 1]])])

        fig, axis = plt.subplots(2, 2)
        Argb = cv.cvtColor(A, cv.COLOR_GRAY2RGB)

        N, M = A.shape[0], A.shape[1]
        dx, dy = 1, 1
        window = 30
        change_begin = np.delete(change_begin, (0), axis=0)
        change_end = np.delete(change_end, (0), axis=0)

        for point in change_begin:
            point[0] = int(point[0] * dx + window / 2)
            point[1] = int(point[1] * dy + window / 2)

        for point in change_end:
            point[0] = int(point[0] * dx + window / 2)
            point[1] = int(point[1] * dy + window / 2)

        axis[0, 0].imshow(Argb)
        axis[0, 0].scatter(change_begin[:, 0], change_begin[:, 1], c='red', s=1)
        axis[0, 0].scatter(change_end[:, 0], change_end[:, 1], c='red', s=1)
        axis[0, 0].set_title("Image and amplitude's points")

        field = change_begin

        nearest_neighbors = NearestNeighbors(n_neighbors=11)
        neighbors = nearest_neighbors.fit(field)
        distances, indices = neighbors.kneighbors(field)
        distances = np.sort(distances[:, 10], axis=0)

        i = np.arange(len(distances))
        knee = KneeLocator(i, distances, S=1, curve='convex', direction='increasing', interp_method='polynomial')

        axis[0, 1].set_title('Knee Point = ' + str(distances[knee.knee]))
        axis[0, 1].plot(knee.x, knee.y, "b", label="data")
        axis[0, 1].vlines(
            knee.knee, plt.ylim()[0], plt.ylim()[1], linestyles="--", label="knee/elbow"
        )
        axis[0, 1].legend(loc="best")

        print(distances[knee.knee])

        eps = 1. * distances[knee.knee]
        samples = 8
        dbscan_cluster = DBSCAN(eps=eps, min_samples=samples)
        dbscan_cluster.fit(field)

        axis[1, 0].scatter(field[:, 0], field[:, 1], c=dbscan_cluster.labels_, s=15)
        axis[1, 0].scatter(change_end[:, 0], change_end[:, 1], c=dbscan_cluster.labels_, s=15)
        axis[1, 0].set_title('DBSCAN eps=' + str(eps) + ' min_samples=' + str(samples))

        labels = dbscan_cluster.labels_.copy()
        uniqlabels, label_count = np.unique(labels, return_counts=True)
        label_count_mean = np.mean(label_count[1:])

        clustered_change_begin = np.array([0, 0], dtype=np.uint32)
        clustered_change_end = np.array([0, 0], dtype=np.uint32)
        filtered_labels = np.empty(0, dtype=np.uint8)
        for i in range(1, uniqlabels.shape[0]):
            if label_count[i] >= label_count_mean:
                clustered_change_begin = np.vstack([clustered_change_begin, field[labels == uniqlabels[i]]])
                clustered_change_end = np.vstack([clustered_change_end,
                                                       change_end[labels == uniqlabels[i]]])
                filtered_labels = np.append(filtered_labels, [i] * label_count[i])

        clustered_change_begin = np.delete(clustered_change_begin, (0), axis=0)
        clustered_change_end = np.delete(clustered_change_end, (0), axis=0)

        axis[1, 1].scatter(clustered_change_begin[:, 0], clustered_change_begin[:, 1], c=filtered_labels, s=10)
        axis[1, 1].scatter(clustered_change_end[:, 0], clustered_change_end[:, 1], c=filtered_labels, s=10)
        axis[1, 1].set_title('[Filtered] DBSCAN eps=' + str(eps) + ' m_samp=' + str(samples))

        plt.show()

        for i in np.unique(filtered_labels):
            x = clustered_change_begin[filtered_labels == i, 0]
            y1 = clustered_change_begin[filtered_labels == i, 1]
            y2 = clustered_change_end[filtered_labels == i, 1]

            lowess = sm.nonparametric.lowess
            z1 = lowess(y1, x, frac=1. / 10)
            z2 = lowess(y2, x, frac=1. / 10)

            plt.plot(z1[:, 0], z1[:, 1], '--', c='red', lw=2)
            plt.plot(z2[:, 0], z2[:, 1], '--', c='red', lw=2)
            plt.fill_between(z1[:, 0], z1[:, 1], z2[:, 1], facecolor='red', alpha=0.1)
            red_square = lns.Line2D([], [], color='red', marker='s', linestyle='None',
                                    markersize=7, label='Переходный слой')
            plt.legend(handles=[red_square])

        plt.imshow(Argb)
        plt.show()

    elif comm == 7:
        print('Add n_components')
        n_comp = int(input())

        predict, means = EM(X=np.expand_dims(pole.flatten(), 1), component=n_comp)
        print('Means: ', means)

        field_predict = np.resize(predict, pole.shape)
        # vizualizateEM(field_predict, n_comp, plot=True)
        field_pr_img = vizualizateEM(field_predict, n_comp, plot=True)

        stratum_class = np.argmin(means)
        intens_stratum = int(stratum_class / n_comp * 255)
        mask_stratum = segments(field_pr_img, intens_stratum, plot=True)

        window = 30
        dx = 1
        dy = 1

        Argb = cv.cvtColor(A[int(window / 2): -int(window / 2) + 1, int(window / 2): -int(window / 2) + 1],
                           cv.COLOR_GRAY2RGB)
        mask_stratum = cv.resize(mask_stratum, np.flip(Argb.shape[:2]), fx=dx, fy=dy,
                                 interpolation=cv.INTER_LINEAR_EXACT)

        layer = np.zeros(Argb.shape, dtype=np.uint8)
        layer[:] = (255, 0, 102)
        # cv.imshow('layer', layer)

        stratum = cv.bitwise_and(Argb, layer, mask=mask_stratum)
        # cv.imshow('stratum', stratum)

        result = cv.addWeighted(Argb, 1, stratum, 0.1, 0.0)

        contours, hierarchy = cv.findContours(mask_stratum.copy(), cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
        cv.drawContours(result, contours, -1, (102, 0, 153), 1, cv.LINE_AA, hierarchy, 3)

        # cv.imshow('result', result)
        # cv.waitKey()
        # cv.destroyAllWindows()

        result = cv.cvtColor(result, cv.COLOR_BGR2RGB)

        cv.imwrite(folder + '/[Result] ' + filename + '_x' + str(dx) + 'y' + str(dy) + 'w' + str(window) + '.jpg',
                   result)

        plt.imshow(result)
        purple_square = lns.Line2D([], [], color=(0.4, 0, 0.6), marker='s', linestyle='None',
                                   markersize=7, label='Переходный слой')
        plt.legend(handles=[purple_square])
        plt.show()




=======
=======
>>>>>>> 4e9401b4c0aba1445f05a0b4f173a62ade31b2c9
            if ind_pick[0].shape[0] > 2:
                ind_pick = np.array([[ind_pick[0][0], ind_pick[0][-1]], [0, 0]])

            field_points = beg_end_pick(ind_pick[0], amplMF, extrMF)

            # перевод точек поля в точки изображения
            N = 657
            window = 30
            dn = 1
            global_points = to_global(field_points, N, window, dn)
            points = np.append(points, global_points)

            M = 1022
            window = 30
            dn = 1
            global_col = to_global([col], M, window, dn)
            global_col = np.resize(global_col, global_points.shape)
            col_of_points = np.append(col_of_points, global_col)

            # if ind_pick[0].shape[0] > 2:
            #    plt.plot(extrMF[1:], amplMF)
            #    plt.title(f"col = {global_col[0]}, ind_pick = {extrMF[[a + 1 for a in ind_pick[0].__array__()]]}\n"
            #              f"fp = {field_points.__array__()}\ngp = {global_points.__array__()}")
            #    plt.show()


        plt.imshow(A, cmap='gray')
        colors = ['yellow', 'red', 'green', 'royalblue', 'violet', 'tomato']
        colors *= int(col_of_points.size / 6) + 1
        j = 0
        for i in range(0, col_of_points.shape[0] - 2, 2):
            plt.scatter(col_of_points[i : i+2], points[i: i+2], c=colors[j], s=1)
            j += 1
        # plt.scatter(col_of_points, points, s=1)
        plt.title("Image and amplitude's points")
        plt.show()

        print("Save analysis? Yes - 1, No - 0")
        temp = int(input())
        if temp == 1:

            Tk().withdraw()
            DIRname = askdirectory()
            print("Add datafile's name for columns\n")
            datafilename = input()
            np.savetxt(DIRname + "/" + datafilename + ".csv", col_of_points, delimiter=";", newline="\n")

            Tk().withdraw()
            DIRname = askdirectory()
            print("Add datafile's name for points\n")
            datafilename = input()
            np.savetxt(DIRname + "/" + datafilename + ".csv", points, delimiter=";", newline="\n")


    elif comm == 6:
        Tk().withdraw()
        uploadfilename = askopenfilename()
        col_of_points = np.loadtxt(uploadfilename, delimiter=";")

        Tk().withdraw()
        uploadfilename = askopenfilename()
        points = np.loadtxt(uploadfilename, delimiter=";")

        plt.imshow(A, cmap='gray')
        colors = ['yellow', 'red', 'green', 'royalblue', 'violet', 'tomato']
        colors *= int(col_of_points.size / 6) + 1
        j = 0
        for i in range(0, col_of_points.shape[0] - 2, 2):
            plt.scatter(col_of_points[i: i + 2], points[i: i + 2], c=colors[j], s=1)
            j += 1
        # plt.scatter(col_of_points, points, s=1)
        plt.title("Image and amplitude's points")
        plt.show()

    elif comm == 7:
        window = 30
        for x, y in zip(col_of_points, points):
            up = int(y) - window
            if up < 0:
                up = 0
            down = int(y) + window
            if down > A.shape[0] - 1:
                down = A.shape[0] - 1
            win_col = A[up: down + 1, int(x)]
            plt.plot(win_col)
            plt.title(f"x = {int(x)}, y from {up} to {down}, center = {int(y)}")
            plt.show()

            '''
            MedianFilterS = signal.medfilt(win_col, kernel_size=11)
            plt.plot(MedianFilterS)
            plt.title(f"Median Filter\nx = {int(x)}, y from {up} to {down}, center = {int(y)}")
            plt.show()
            '''

            amplWin, extrWin, maskWin = find_extr_int(win_col)
            plt.plot(extrWin[1:], amplWin)
            plt.title(f"Amplitudes\nx = {int(x)}, y from {up} to {down}, center = {int(y)}")
            plt.show()

'''
<<<<<<< HEAD
>>>>>>> 4e9401b4c0aba1445f05a0b4f173a62ade31b2c9
=======
>>>>>>> 4e9401b4c0aba1445f05a0b4f173a62ade31b2c9
        #амплитуды столбца поля
        #plt.plot(np.gradient(pole[:, col]))
        #plt.show()
        peaks_max = signal.find_peaks(pole[:, col])
        peaks_min = signal.find_peaks((-1.0)*pole[:, col])
        #ind_max = signal.argrelextrema(pole[:, 0], np.greater)
        #ind_min = signal.argrelextrema(pole[: , 0], np.less)
        if peaks_max[0][0] < peaks_min[0][0] and peaks_max[0][0] != 0:
            peaks_min1 = np.insert(peaks_min[0], 0, 0)
            peaks_max1 = peaks_max[0]
        elif peaks_max[0][0] > peaks_min[0][0] and peaks_min[0][0] != 0:
            peaks_max1 = np.insert(peaks_max[0], 0, 0)
            peaks_min1 = peaks_min[0]

        if peaks_max1[-1] < peaks_min1[-1] and peaks_min1[-1] != (pole.shape[0] - 1):
            peaks_max1 = np.concatenate((peaks_max1, [pole.shape[0] - 1]))
        elif peaks_max1[-1] > peaks_min1[-1] and peaks_max1[-1] != (pole.shape[0] - 1):
            peaks_min1 = np.concatenate((peaks_min1, [pole.shape[0] - 1]))

        ampl = np.empty(0, dtype=numpy.float64)
        if peaks_max1[0] > peaks_min1[0]:
            for i in range(min(peaks_min1.size, peaks_max1.size)):
                if (i + 1 != min(peaks_min1.size, peaks_max1.size)):
                    ampl = np.append(ampl, [pole[peaks_max1[i], col] - pole[peaks_min1[i], col],
                                            pole[peaks_max1[i], col] - pole[peaks_min1[i + 1], col]])

        if peaks_max1[0] < peaks_min1[0]:
            for i in range(min(peaks_min1.size, peaks_max1.size)):
                if i + 1 != min(peaks_min1.size, peaks_max1.size):
                        ampl = np.append(ampl, [pole[peaks_max1[i], col] - pole[peaks_min1[i], col],
                                                pole[peaks_max1[i + 1], col] - pole[peaks_min1[i], col]])

        plt.plot(ampl)
        plt.show()

        #Медианная оценка и её амплитуды
        pG = MedianFilter
        peaks_maxG = signal.find_peaks(pG)
        peaks_minG = signal.find_peaks((-1.0) * pG)
        if peaks_maxG[0][0] < peaks_minG[0][0] and peaks_maxG[0][0] != 0:
            peaks_min1G = np.insert(peaks_minG[0], 0, 0)
            peaks_max1G = peaks_maxG[0]
        elif peaks_maxG[0][0] > peaks_minG[0][0] and peaks_minG[0][0] != 0:
            peaks_max1G = np.insert(peaks_maxG[0], 0, 0)
            peaks_min1G = peaks_minG[0]

        if peaks_max1G[-1] < peaks_min1G[-1] and peaks_min1G[-1] != (pG.shape[0] - 1):
            peaks_max1G = np.concatenate((peaks_max1G, [pG.shape[0] - 1]))
        elif peaks_max1G[-1] > peaks_min1G[-1] and peaks_max1G[-1] != (pG.shape[0] - 1):
            peaks_min1G = np.concatenate((peaks_min1G, [pG.shape[0] - 1]))

        amplG = np.empty(0, dtype=numpy.float64)
        if peaks_max1G[0] > peaks_min1G[0]:
            for i in range(min(peaks_min1G.size, peaks_max1G.size)):
                #amplG = np.append(amplG, [(pG[peaks_max1G[i]] - pG[peaks_min1G[i]]) / abs(peaks_max1G[i] - peaks_min1G[i]),
                #                             (pG[peaks_max1G[i]] - pG[peaks_min1G[i + 1]] / abs(peaks_max1G[i] - peaks_min1G[i + 1]))])
                amplG = np.append(amplG, [pG[peaks_max1G[i]] - pG[peaks_min1G[i]],
                                          pG[peaks_max1G[i]] - pG[peaks_min1G[i + 1]]])

        plt.plot(amplG)
        plt.show()

        amplGG = signal.medfilt(amplG, kernel_size=7)
        plt.plot(amplGG)
        plt.show()

test = [1, 2, 3, 4, 5, 6, 7]
with open("C:/Users/bortnikov.2018/Desktop/JPEG/testaaaaa.csv", mode="w+", encoding='utf-8') as w_file:
    file_writer = csv.writer(w_file, delimiter=';', lineterminator='\n')
    for i in range(len(test)):
        file_writer.writerow((test[i],))
        
        
name = 'test 4'
win_size = 30
dx = 3
dy = 3

#input and convert image
imgorig = matplotlib.pyplot.imread("C:/Users/bortnikov.2018/Desktop/JPEG/Контрольные тесты/" + name + ".jpg")
A = np.delete(imgorig, (1, 2), 2)
A = np.resize(A, (imgorig.shape[0], imgorig.shape[1]))
b = np.zeros((win_size, win_size))
windows = roll(A, b, dx, dy) #array of windows

e = np.array([win_size, int((win_size + 1) / 2), int((win_size + 3) / 4)])

cv.imshow(name, windows[0][0])
cv.waitKey()
cv.destroyAllWindows()


<<<<<<< HEAD
<<<<<<< HEAD
#pole = _method_cubes(windows, e)
pole = _method_prism(windows, e)
=======
#pole = method_cubes(windows, e)
pole = method_prism(windows, e)
>>>>>>> 4e9401b4c0aba1445f05a0b4f173a62ade31b2c9
=======
#pole = method_cubes(windows, e)
pole = method_prism(windows, e)
>>>>>>> 4e9401b4c0aba1445f05a0b4f173a62ade31b2c9

plt.hist(pole)
plt.show()



imgout = np.empty((windows.shape[0], windows.shape[1]), dtype=numpy.int8)
temp = np.empty((windows.shape[0], windows.shape[1]))


with open("C:/Users/bortnikov.2018/Desktop/JPEG/Контрольные тесты/" + name + ' ' + str(win_size) + ".csv", mode="a+", encoding='utf-8') as w_file:
    file_writer = csv.writer(w_file, delimiter=';', lineterminator='\n')
    #file_writer.writerow((win_size,))
    p = pole.flatten()
    for elem in p:
        file_writer.writerow((elem,))



with open("C:/Users/bortnikov.2018/Desktop/JPEG/" + name + ".csv", encoding='utf-8') as file_r:
    file_reader = csv.reader(file_r, delimiter=';')
    i = 0
    for row in file_reader:
        temp[i] = [float(j) for j in row]
        i += 1



for i in range(imgout.shape[0]):
    for j in range(imgout.shape[1]):
        #if pole[i][j] < 3:
        #    imgout[i][j] = 0
        #else:
            imgout[i][j] = int(255 * pole[i][j] / 4)


cv.imshow(name, imgout)
cv.waitKey()
cv.destroyAllWindows()
<<<<<<< HEAD
<<<<<<< HEAD


elif comm == 8:
    eps = 0.0
    N1 = 0
    sum1 = 0.0
    N2 = 0
    sum2 = 0.0
    for i in range(0, len(points), 4):
        temp1 = points[i + 1] - points[i]
        if temp1 > eps:
            sum1 += temp1
            N1 += 1
        temp2 = points[i + 3] - points[i + 2]
        if temp2 > eps:
            sum2 += temp2
            N2 += 1
    sum1 /= N1
    sum2 /= N2
    print('Write scale in µm')
    scale = int(input())
    width1 = sum1 / 140 * scale
    width2 = sum2 / 140 * scale
    print(width1)
    print(width2)


def showsegmentpole3(pole):
    imgout = np.empty((pole.shape[0], pole.shape[1]), dtype=numpy.uint8)
    for i in range(imgout.shape[0]):
        for j in range(imgout.shape[1]):
            if pole[i][j] < 2.0:
                imgout[i][j] = int(250)
            # elif pole[i][j] < 2.05:
            #    imgout[i][j] = int(0)
            elif pole[i][j] < 2.60:
                imgout[i][j] = int(0)
            # elif pole[i][j] < 2.61:
            #    imgout[i][j] = int(100)
            # elif pole[i][j] < 2.46:
            #    imgout[i][j] = int(100)
            # elif pole[i][j] < 2.72:
            #    imgout[i][j] = int(150)
            elif pole[i][j] < 3.0:
                imgout[i][j] = int(180)
            else:
                imgout[i][j] = int(250)
    plt.imshow(imgout, cmap='gray')
    plt.show()
    
    
def find_extr_int(data):
    mask = np.zeros(data.size, dtype=numpy.int8)
    extr = np.empty(0, dtype=numpy.uint32)
    ampl = np.empty(0, dtype=numpy.float64)
    i = 0
    toMax = False
    while data[i + 1] == data[i] and i + 1 != data.size:
        i += 1
    extr = np.append(extr, [i])
    if i + 1 != data.size:
        if data[i + 1] > data[i]:
            mask[i] = -1
            toMax = True
        else:
            mask[i] = 1
            toMax = False
    while i + 1 != data.size:
        if toMax == True:
            while i + 1 != data.size and data[i + 1] >= data[i]:
                i += 1
            mask[i] = 1
            toMax = False
        else:
            while i + 1 != data.size and data[i + 1] <= data[i]:
                i += 1
            mask[i] = - 1
            toMax = True
        ampl = np.append(ampl, [abs(int(data[i]) - int(data[extr[-1]]))])
        extr = np.append(extr, [i])
    return ampl, extr, mask

def points_pick(ind_pick, extrMF, col):
    points = np.array([0, 0], dtype=np.uint32)
    for i in ind_pick:
        j = i.copy()
        points = np.vstack([points, np.array([col, extrMF[j]])])
    points = np.delete(points, (0), axis=0)
    return points
    
def beg_end_pick(ind_pick, amplMF, extrMF):
    points = np.empty(0, dtype=numpy.uint32)
    for i in ind_pick:
        j = i.copy()
        # while amplMF[i] / amplMF[j] < 4 and j > 0:
        #    j -= 1
        points = np.append(points, [extrMF[j]])
        points = np.append(points, [extrMF[j + 1]])
        # j = i.copy()
        # while amplMF[i] / amplMF[j] < 4 and j < amplMF.shape[0] - 1:
        #    j += 1

    return points


def to_global(points, N, window, dn):
    new_points = np.empty(0, dtype=numpy.uint32)
    if points.size >= 2:
        for i in range(0, points.size, 2):
            x1 = int(points[i] * dn + window / 2)
            x2 = int(points[i + 1] * dn + window / 2)
            if N - window / 2 > x2 != x1 and x1 > 0 and points[0] > 0:
                new_points = np.append(new_points, [x1, x2])
    elif points.size == 1:
        p = int(points[0] * dn + window / 2)
        new_points = np.append(new_points, [p])
    return new_points


def to_global_field(change_begin, N, M, window, dx, dy):
    for point in change_begin:
        point[1] = int(points[1] * dx + window / 2)
        point[0] = int(points[0] * dy + window / 2)
    return change_begin
    
  
def Prewitt(grayImage):
    # Оператор Prewitt
    kernelx = np.array([[1, 1, 1], [0, 0, 0], [-1, -1, -1]], dtype=int)
    kernely = np.array([[-1, 0, 1], [-1, 0, 1], [-1, 0, 1]], dtype=int)
    x = cv.filter2D(grayImage, cv.CV_16S, kernelx)
    # plt.imshow(x, cmap=plt.cm.gray)
    # plt.show()
    y = cv.filter2D(grayImage, cv.CV_16S, kernely)
    # plt.imshow(y, cmap=plt.cm.gray)
    # plt.show()
    # Turn uint8
    absX = cv.convertScaleAbs(x)
    # plt.imshow(absX, cmap=plt.cm.gray)
    # plt.show()
    absY = cv.convertScaleAbs(y)
    # plt.imshow(absY, cmap=plt.cm.gray)
    # plt.show()
    Prewitt_ = cv.addWeighted(absX, 1.0, absY, 1.0, 0)
    plt.imshow(Prewitt_, cmap=plt.cm.gray)
    plt.title('Оператор Превитта')
    plt.show()


def Sobel(grayImage):
    # Оператор Собеля
    x = cv.Sobel(grayImage, cv.CV_16S, 1, 0)  # Найдите первую производную x
    y = cv.Sobel(grayImage, cv.CV_16S, 0, 1)  # Найдите первую производную y
    absX = cv.convertScaleAbs(x)
    absY = cv.convertScaleAbs(y)
    Sobel_ = cv.addWeighted(absX, 0.5, absY, 0.5, 0)
    plt.imshow(Sobel_, cmap=plt.cm.gray)
    plt.title('Оператор Собеля')
    plt.show()


def Laplacian(grayImage):
    # Алгоритм Лапласа
    dst = cv.Laplacian(grayImage, cv.CV_16S, ksize=3)
    Laplacian_ = cv.convertScaleAbs(dst)
    plt.imshow(Laplacian_, cmap=plt.cm.gray)
    plt.title('Оператор Лапласа')
    plt.show()


def p_gauss(x, iS, dS, mu):
    # n = x.shape[0]
    n = x.size
    return np.exp(np.dot(np.dot(-0.5 * (x.T - mu.T), iS), (x - mu))) / sqrt((2 * math.pi) ** n * dS)


def p_mix(x, iS, dS, mu, w):
    p = 0
    # n = x.shape[0]
    n = x.shape
    k = iS.shape[0]
    for j in range(0, int(k)):
        # p += w[j] * p_gauss(x, iS[j, :, :], dS[j], mu[j, :])
        p += w[j] * p_gauss(x, iS[j], dS[j], mu[j])
    return p


def eigsorted(cov):
    vals, vecs = np.linalg.eigh(cov)
    order = vals.argsort()[::-1]
    return vals[order], vecs[:, order]


def EMwithAdding():
    X = np.loadtxt('vec.txt')
    X = np.reshape(X, (-1, 1))
    m = X.shape[0]
    n = X.ndim

    R = 0.9
    MAX_ITER = 7
    m0 = round(m / 3)

    k = 1
    w = np.array([1], dtype=np.float64)

    mu = np.array([np.mean(X, axis=0)], dtype=np.float64)
    S = np.array([[np.cov(X.T)]], dtype=np.float64)
    iS = np.linalg.inv(S)
    dS = np.array(np.linalg.det(S), dtype=np.float64)
    dS = S.copy()

    for iter in range(1, MAX_ITER):
        p = np.empty(0, dtype=np.float64)
        for i in range(m):
            x = np.array(X[i])
            p = np.append(p, [p_mix(x, iS, dS, mu, w)])
        mp = np.amax(p)

        p = p - R * mp
        p = np.abs(p) - p

        low_p_idx = np.asarray(np.nonzero(p)[0], dtype=np.int)
        low_p_size = low_p_idx.shape[0]

        U = X[low_p_idx, :]
        # U = X[low_p_idx]

        if (low_p_size < m0):
            break

        k += 1
        R *= R

        # cU = np.array(np.cov(U.T))
        cU = np.reshape(np.cov(U.T), (-1, 1))
        wk = U.shape[0] / m

        muU = np.mean(U, axis=0) * wk

        means = np.vstack([mu, muU])
        covs = np.append(S, cU)
        covs = np.reshape(covs, (k, 1, 1))
        precis = np.linalg.inv(covs)
        # precis = np.power(covs, -1)
        weights = np.append([w * (1 - wk)], [wk])

        model = GMM(n_components=k,
                    covariance_type='full',
                    # covariance_type='diag',
                    means_init=means,
                    precisions_init=precis,
                    weights_init=weights,
                    tol=10e-4,
                    max_iter=50
                    )
        model.fit(X)
        S = model.covariances_
        iS = np.linalg.inv(S)
        # iS = np.power(S, -1)
        dS = np.linalg.det(S)
        # dS = S.copy()
        mu = model.means_
        w = model.weights_


        ax = plt.gca()
        plt.scatter(X[:,0], X[:, 1])
        plt.scatter(mu[:,0], mu[:, 1], c='red')

        for j in range(k):
            vals, vecs = eigsorted(S[j])
            theta = np.degrees(np.arctan2(*vecs[:, 0][::-1]))
            width, height = 2 * 3 * np.sqrt(vals)
            el = ptc.Ellipse(xy=(mu[j,0], mu[j, 1]), width=width, height=height, angle=theta,
                             color='green', fill=False, linewidth=1)
            ax.add_patch(el)


        plt.show()"""
=======
'''
>>>>>>> 4e9401b4c0aba1445f05a0b4f173a62ade31b2c9
=======
'''
>>>>>>> 4e9401b4c0aba1445f05a0b4f173a62ade31b2c9
