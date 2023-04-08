import numpy as np
import cv2 as cv
from FunctionsDistance import functions_distance_normal
from metadata_search import getScale
from analys_data import analys_width
import matplotlib.pyplot as plt

path2_upper = r'C:\Users\bortn\Desktop\Diplomchichek\Тесты\5-11\2-upper.jpg'
path2_lower = r'C:\Users\bortn\Desktop\Diplomchichek\Тесты\5-11\2-lower.jpg'
path1_upper = r'C:\Users\bortn\Desktop\Diplomchichek\Тесты\5-11\1-upper.jpg'
path1_lower = r'C:\Users\bortn\Desktop\Diplomchichek\Тесты\5-11\1-lower.jpg'
original = r'C:\Users\bortn\Desktop\diplomchik\analysis\new dataset\5\1-11\1-11.tif'

list1 = None
i = 0
paths_upper = [path2_upper, path1_upper]
paths_lower = [path2_lower, path1_lower]
for u, l in zip(paths_upper, paths_lower):
    f = open(u, "rb")
    chunk = f.read()
    chunk_arr = np.frombuffer(chunk, dtype=np.uint8)
    img_upper = cv.imdecode(chunk_arr, cv.IMREAD_GRAYSCALE)
    f.close()

    f = open(l, "rb")
    chunk = f.read()
    chunk_arr = np.frombuffer(chunk, dtype=np.uint8)
    img_lower = cv.imdecode(chunk_arr, cv.IMREAD_GRAYSCALE)
    f.close()

    shapeX = img_lower.shape[0]

    _, img_upper = cv.threshold(img_upper, 253, 255, cv.THRESH_BINARY)
    _, img_lower = cv.threshold(img_lower, 253, 255, cv.THRESH_BINARY)

    coor_upper = np.unique(np.c_[np.nonzero(img_upper)], axis=0)
    coor_lower = np.unique(np.c_[np.nonzero(img_lower)], axis=0)

    if np.min(np.sort(coor_upper[:, 1]) == np.sort(np.unique(coor_upper[:, 1]))):
        print('По оси Y нет одинаковый значений - всё хорошо')
    else:
        print('По оси Y есть одинаковые значения - всё плохо (или нет?)')

    coor_upper = coor_upper[coor_upper[:, 1].argsort()]
    coor_lower = coor_lower[coor_lower[:, 1].argsort()]

    coor_upper[:, [1, 0]] = coor_upper[:, [0, 1]]
    coor_lower[:, [1, 0]] = coor_lower[:, [0, 1]]

    coor_upper[:, 1] = shapeX - coor_upper[:, 1]
    coor_lower[:, 1] = shapeX - coor_lower[:, 1]

    dList = functions_distance_normal(coor_lower, coor_upper)
    if i == 0:
        list1 = dList
        i += 1
    else:
        list1 = np.concatenate((list1, dList), axis=0)
    #ScaleCoef = getScale(original)


plt.hist(list1 * getScale(original), bins=25, facecolor="#BEF73E", edgecolor='none')
plt.grid()
plt.show()
print('Медиана: ', np.median(list1) * getScale(original))
print('Среднее арифметическое: ', np.mean(list1) * getScale(original))
# analys_width(dList, ScaleCoef)