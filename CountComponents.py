import numpy as np
from sklearn.mixture import GaussianMixture
import matplotlib.pyplot as plt
import scipy.stats as stats
from math import sqrt

class CountComponents:
    colors = {'0': '#BEF73E',
              '1': '#C9007A',
              '2': '#FF7F00',
              '3': '#0772A1',
              '4': '#2818B1'}

    lineStyles = ['-', '--']

    def __init__(self, minCount:int, maxCount:int, showPlot:bool=False):
        if 0 < minCount <= maxCount:
            self._minCount = minCount
            self._maxCount = maxCount
        else:
            raise ValueError("Incorrect value of the possible number of components")
        self._showPlot = showPlot

    def __call__(self, data:np.ndarray) -> int:
        # counts, bins, _ = plt.hist(data, bins=250, facecolor=CountComponents.colors['0'],
        #                                edgecolor='none', density=True, label='ПФР')
        maxValue = np.amax(data)
        minValue = np.amin(data)
        step = (maxValue - minValue) / 250
        bins = [minValue + step * i for i in range(250)]
        counts = np.bincount(np.digitize(data, bins[1:])) / data.shape[0] / step
        # frequency = counts / data.shape[0] / step
        # counts = frequency
        bins = np.asarray(bins)

        if self._showPlot:
            plt.hist(data, bins, facecolor=CountComponents.colors['0'], edgecolor='none', density=True, label='ПФР')
            plt.scatter(bins, counts, s=3, c='red', marker='x')

        chiSquares = []
        for curCount in range(self._minCount, self._maxCount + 1):
            X = np.expand_dims(data, 1)
            model = GaussianMixture(n_components=curCount, covariance_type='full')
            model.fit(X)
            mu = model.means_
            sigma = model.covariances_
            x = bins
            # x = (bins[1:] + bins[:-1]) / 2.
            densityFuncValues = np.zeros(x.shape)
            for i in range(curCount):
                densityFuncValues += model.weights_[i] * stats.norm.pdf(x, mu[i][0], sqrt(sigma[i][0][0]))

            if self._showPlot:
                plt.plot(x, densityFuncValues, color=CountComponents.colors[str(curCount)], label=rf'$f_{curCount}(x)$', linewidth=2,
                         linestyle=CountComponents.lineStyles[curCount - self._minCount])

            chi, _ = stats.chisquare(counts, f_exp=densityFuncValues)
            chiSquares.append(chi)

        if self._showPlot:
            col_labels = [r'$\it{n}$', r'$\chi^2$']
            table_vals = [[curCount, f'{chiSquares[curCount - self._minCount]:.2f}']
                          for curCount in range(self._minCount, self._maxCount + 1)]
            table = plt.table(cellText=table_vals, colLabels=col_labels, loc='center right',
                              colWidths=[0.05, 0.2], cellLoc='center')
            table.set_zorder(100)
            table.set_fontsize(14)
            table.scale(1, 2)

            plt.grid(True)
            plt.legend(fontsize=14)
            plt.xlabel(r'Фрактальная размерность, $\it{D}$')
            plt.ylabel('Частота')
            plt.tight_layout()
            plt.show()

        return chiSquares.index(min(chiSquares)) + self._minCount