from scipy import stats
import statistics
import numpy as np
import matplotlib.pyplot as plt


def analys_width(widthList, ScaleCoef, nBins=15):
    quartiles = statistics.quantiles(widthList)

    print('С выбросами')
    print('Среднее арифметическое: ', np.mean(widthList) * ScaleCoef)
    print('Медиана: ', np.median(widthList) * ScaleCoef)
    #mode, _ = stats.mode(widthList, keepdims=True)
    #print('Мода1: ', mode[0] * ScaleCoef)
    # forMode =
    # mode2, _ = stats.mode(np.around(forMode, decimals=8), keepdims=True)
    # print('Мода2: ', mode2[0])
    counts, bins, _ = plt.hist(widthList * ScaleCoef, bins=nBins)
    mode = (bins[np.argmax(counts) + 1] - (bins[np.argmax(counts)])) / 2 + bins[np.argmax(counts)]
    print('Мода: ', mode)
    print('Квартили: ', np.array(quartiles) * ScaleCoef)

    print('\n')

    coef = 1.5
    maxLimin = quartiles[2] + coef * (quartiles[2] - quartiles[0])
    minLimin = quartiles[0] - coef * (quartiles[2] - quartiles[0])

    without_outlier = widthList[np.logical_and(minLimin <= widthList, widthList <= maxLimin)]
    print('Без выбросов')
    print('Среднее арифметическое: ', np.mean(without_outlier) * ScaleCoef)
    print('Медиана: ', np.median(without_outlier) * ScaleCoef)
    counts, bins, _ = plt.hist(without_outlier * ScaleCoef, bins=nBins)
    #print(counts)
    #print(bins)
    mode = (bins[np.argmax(counts) + 1] - (bins[np.argmax(counts)])) / 2 + bins[np.argmax(counts)]
    print('Мода: ', mode)
    plt.close()
    # mode, _ = stats.mode(without_outlier, keepdims=True)
    # print('Мода: ', mode[0] * ScaleCoef)
    # forMode = without_outlier * ScaleCoef
    # mode2, _ = stats.mode(np.around(forMode, decimals=8), keepdims=True)
    # print('Мода2: ', mode2[0])
    print('Квартили: ', np.array(statistics.quantiles(without_outlier)) * ScaleCoef)
    print('\n')

    fig, axis = plt.subplot_mosaic([['plot_with', 'hist_with', 'boxplot'],
                                    ['plot_without', 'hist_without', 'boxplot']])
    axis['plot_with'].plot(widthList * ScaleCoef)
    axis['plot_with'].set_title('Значения ширины вдоль контура')
    axis['plot_with'].set_ylim(bottom=0, top=4 * 1e-5)
    axis['hist_with'].hist(widthList * ScaleCoef, bins=nBins, facecolor="#BEF73E", edgecolor='none')
    axis['hist_with'].set_title('Гистограмма значений')
    axis['hist_with'].set_ylim(bottom=0, top=290)
    axis['hist_with'].grid()

    axis['plot_without'].plot(without_outlier * ScaleCoef)
    axis['plot_without'].set_title('Значения ширины вдоль контура (без выбросов)')
    axis['plot_without'].set_ylim(bottom=0, top=4 * 1e-5)
    axis['hist_without'].hist(without_outlier * ScaleCoef, bins=nBins, facecolor="#BEF73E", edgecolor='none')
    axis['hist_without'].set_title('Гистограмма значений (без выбросов)')
    axis['hist_without'].set_ylim(bottom=0, top=290)
    axis['hist_without'].grid()

    axis['boxplot'].boxplot(widthList)
    axis['boxplot'].set_title('Диаграмма размаха')
    plt.show()