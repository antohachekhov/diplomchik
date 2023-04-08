import numpy as np
from metadata_search import getScale
import matplotlib.pyplot as plt
from analys_data import analys_width

nFiles = 3

original = r'C:\Users\bortn\Desktop\diplomchik\analysis\new dataset\5\1-11\1-11.tif'

if nFiles == 2:
    list1path = r'C:\Users\bortn\Desktop\diplomchik\listDistances_1.txt'
    list2path = r'C:\Users\bortn\Desktop\diplomchik\listDistances_2.txt'

    list1 = np.loadtxt(list1path)
    list2 = np.loadtxt(list2path)

    list = np.concatenate((list1, list2), axis=0)
elif nFiles == 1:
    listpath = r'C:\Users\bortn\Desktop\diplomchik\listDistances_0.txt'
    list = np.loadtxt(listpath)
elif nFiles == 3:
    list0path = r'C:\Users\bortn\Desktop\diplomchik\listDistances_0.txt'
    list1path = r'C:\Users\bortn\Desktop\diplomchik\listDistances_1.txt'
    list2path = r'C:\Users\bortn\Desktop\diplomchik\listDistances_2.txt'
    list0 = np.loadtxt(list0path)
    list1 = np.loadtxt(list1path)
    list2 = np.loadtxt(list2path)
    list = np.concatenate((list0, list1, list2), axis=0)

# analys_width(list, getScale(original))

plt.hist(list * getScale(original), bins=15, facecolor="#BEF73E", edgecolor='none')
plt.grid()
plt.show()
print('Медиана: ', np.median(list) * getScale(original))
print('Среднее арифметическое: ', np.mean(list) * getScale(original))

