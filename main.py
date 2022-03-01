import csv
import matplotlib.pyplot
import numpy
import numpy as np
from math import ceil, sqrt, log
import matplotlib.pyplot as plt
import cv2 as cv
from tkinter import Tk
from tkinter.filedialog import askopenfilename, askdirectory
from scipy import signal


# creater of windows
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


def method_cubes(windows, e):
    print('count = ', windows.shape[0], ' : ', windows.shape[1])
    for i in range(windows.shape[0]):
        for j in range(windows.shape[1]):
            t_win = windows[i][j]
            print('window ', i, ' : ', j)
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
            # lgre = np.zeros(e.shape)
            for k in range(e.shape[0]):
                lgn[k] = log(n[k])
                lge[k] = log(e[k])
            A = np.vstack([lge, np.ones(len(lge))]).T
            D, c = np.linalg.lstsq(A, lgn, rcond=None)[0]
            print(-D)
            pole[i][j] = -D
    return pole


def method_prism(windows, s):
    print('count = ', windows.shape[0], ' : ', windows.shape[1])
    for i in range(windows.shape[0]):
        for j in range(windows.shape[1]):
            t_win = windows[i][j]
            print('window ', i, ' : ', j)
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

            lgS = np.zeros(Se.shape)
            lge = np.zeros(s.shape)
            for k in range(s.shape[0]):
                lgS[k] = log(Se[k])
                lge[k] = log(s[k])
            # plt.plot(lge, lgS)
            # plt.show()
            A = np.vstack([lge, np.ones(len(lge))]).T
            D, Cc = np.linalg.lstsq(A, lgS, rcond=None)[0]
            D2 = 2 - D
            print(D1)
            print(D2)
            pole[i][j] = D2
    return pole


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
    dy = int(input())
    # dx, dy = 3, 3
    b = np.zeros((win_size, win_size))
    windows = roll(A, b, dx, dy)  # array of windows
    e = np.array([win_size, int((win_size + 1) / 2), int((win_size + 3) / 4)])
    pole = np.zeros((windows.shape[0], windows.shape[1]))

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

    print("All done!")
elif comm == 2:
    Tk().withdraw()
    uploadfilename = askopenfilename()
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

            # медианная оценка выборки
            'ВНИМАНИЕ! Параметр - окно медианного фильтра'
            MedianFilter = signal.medfilt(column, kernel_size=21)
            # plt.plot(MedianFilter)
            # plt.show()

            # амплитуда оценки
            amplMF, extrMF, maskMF = find_extr(MedianFilter)
            # plt.plot(extrMF[1:], amplMF)
            # plt.show()

            # определение пиков амплитудов
            'ВНИМАНИЕ! Параметр - высота амплитуды перехода'
            ind_pick = np.where(amplMF > 0.4)
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


#pole = method_cubes(windows, e)
pole = method_prism(windows, e)

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
'''
