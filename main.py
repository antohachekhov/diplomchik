import csv
import matplotlib.pyplot
import numpy
import numpy as np
import math
import matplotlib.pyplot as plt
import cv2 as cv


def fractal_dimension(Z, threshold=0.9):

    # Only for 2d image
    print(Z.shape)
    A = np.delete(Z, (1,2), 2)
    A = np.resize(A, (659, 1023))
    Z = A
    assert(len(Z.shape) == 2)

    # From https://github.com/rougier/numpy-100 (#87)
    def boxcount(Z, k):
        S = np.add.reduceat(
            np.add.reduceat(Z, np.arange(0, Z.shape[0], k), axis=0),
                               np.arange(0, Z.shape[1], k), axis=1)

        # We count non-empty (0) and non-full boxes (k*k)
        return len(np.where((S > 0) & (S < k*k))[0])


    # Transform Z into a binary array
    Z = (Z < threshold)

    # Minimal dimension of image
    p = min(Z.shape)

    # Greatest power of 2 less than or equal to p
    n = 2**np.floor(np.log(p)/np.log(2))

    # Extract the exponent
    n = int(np.log(n)/np.log(2))

    # Build successive box sizes (from 2**n down to 2**1)
    sizes = 2**np.arange(n, 1, -1)

    # Actual box counting with decreasing size
    counts = []
    for size in sizes:
        counts.append(boxcount(Z, size))

    # Fit the successive log(sizes) with log (counts)
    coeffs = np.polyfit(np.log(sizes), np.log(counts), 1)
    return -coeffs[0]

#I = matplotlib.pyplot.imread("C:/Users/bortnikov.2018/Desktop/JPEG/b.jpg")/256.0
#print("Minkowski–Bouligand dimension (computed): ", fractal_dimension(I))
#print("Haussdorf dimension (theoretical):        ", (np.log(3)/np.log(2)))

def roll(a,      # ND array
         b,      # rolling 2D window array
         dx,   # horizontal step, abscissa, number of columns
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
    pole = np.zeros((windows.shape[0], windows.shape[1]))
    print('count = ', windows.shape[0], ' : ', windows.shape[1])
    for i in range(windows.shape[0]):
        for j in range(windows.shape[1]):
            t_win = windows[i][j]
            print('window ', i, ' : ', j)
            n = np.zeros(e.shape)
            for ei in range(e.shape[0]):
                size = e[ei]
                n_e = 0
                n_x = math.ceil(t_win.shape[1]/size)
                n_y = math.ceil(t_win.shape[0] / size)
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
                            n_e += math.ceil(max / size) - math.ceil(min / size) + 1
                        else:
                            n_e += 1
                n[ei] = n_e
            lgn = np.zeros(n.shape)
            lge = np.zeros(e.shape)
            #lgre = np.zeros(e.shape)
            for k in range(e.shape[0]):
                lgn[k] = math.log(n[k])
                lge[k] = math.log(e[k])
                '''
                if e[k] == 1:
                    lgre[k] = math.log(1.0 / e[k-1])
                else:
                    lgre[k] = math.log(1.0 / e[k])
                    '''
            A = np.vstack([lge, np.ones(len(lge))]).T
            D, c = np.linalg.lstsq(A, lgn, rcond=None)[0]
            print(-D)
            pole[i][j] = -D
    return pole




name = 'test 4'
win_size = 25
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


pole = method_cubes(windows, e)
plt.hist(pole)
plt.show()


imgout = np.empty((windows.shape[0], windows.shape[1]), dtype=numpy.int8)
temp = np.empty((windows.shape[0], windows.shape[1]))


'''with open("C:/Users/bortnikov.2018/Desktop/JPEG/" + name + ".csv", mode="w", encoding='utf-8') as w_file:
    file_writer = csv.writer(w_file, delimiter=';')
    file_writer.writerows(pole)




with open("C:/Users/bortnikov.2018/Desktop/JPEG/" + name + ".csv", encoding='utf-8') as file_r:
    file_reader = csv.reader(file_r, delimiter=';')
    i = 0
    for row in file_reader:
        temp[i] = [float(j) for j in row]
        i += 1

'''
for i in range(imgout.shape[0]):
    for j in range(imgout.shape[1]):
        imgout[i][j] = int(255 * pole[i][j] / 4)


cv.imshow(name, imgout)
cv.waitKey()
cv.destroyAllWindows()