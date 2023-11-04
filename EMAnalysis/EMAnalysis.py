import numpy as np
import scipy.stats as stats
from sklearn.mixture import GaussianMixture
from matplotlib import pyplot as plt
from math import sqrt
from matplotlib.patches import Rectangle

colors = {'0': '#BEF73E',
          '1': '#C9007A',
          '2': '#FF7F00',
          '3': '#0772A1',
          '4': '#2818B1'}

colorsList = ['red', 'blue']

filedFile = r"C:\Users\bortn\Desktop\diplomchik\analysis\new dataset\5\1-11\field_filt_prism_w30x1y1.csv"
#filedFile = r"C:\Users\bortn\Desktop\diplomchik\analysis\new dataset\20\1-04\field_field_prism_w30x1y1.csv"
field = np.loadtxt(filedFile, delimiter=';')

comp_array = [2, 3]
chisquares = []

# plt.title('Распределение фрактальных размерностей на изображении')

# вывод гистограммы распределений фрактальных размерностей в поле
# count - кол-во, bins - значение размерности
counts, bins, _ = plt.hist(field.flatten(), bins=250, facecolor=colors['0'],
                           edgecolor='none', density=True, label='ПФР')

#create legend
#handles = [Rectangle((0, 0), 1, 1, color=colors['0'], ec="k")]
#labels = ["ПФР"]
#plt.legend(handles, labels)

for comp in comp_array:
    X = np.expand_dims(field.flatten(), 1)
    model = GaussianMixture(n_components=comp, covariance_type='full')
    model.fit(X)
    mu = model.means_
    sigma = model.covariances_

    # Прорисовка функции плотности для каждой компоненты
    # for i in range(comp):
    #     xi = np.linspace(1.5, 3.5, 600)
    #     plt.plot(xi, model.weights_[i] * stats.norm.pdf(xi, mu[i][0], sqrt(sigma[i][0][0])),
    #              label=rf'$\mu_{i}={mu[i][0]:.6f}, \sigma_{i}$={sigma[i][0][0]:.6f}', linewidth=2)

    x = (bins[1:] + bins[:-1]) / 2.
    res = np.zeros(x.shape)
    for i in range(comp):
        res += model.weights_[i] * stats.norm.pdf(x, mu[i][0], sqrt(sigma[i][0][0]))
    # for i in range(comp):
    #     leg += '' #rf'$\mu_{i}={mu[i][0]:.2f}, \sigma_{i}$={sigma[i][0][0]:.2f}' + '\n'
    plt.plot(x, res, color=colors[str(comp)], label=rf'$f_{comp}(x)$', linewidth=2)

    # создание таблицы с параметрами компонент в смеси
    # col_labels = [r'$\it{i}$', r'$\omega_i$', r'$\mu_i$', r'$\sigma_i$']
    # table_vals = [[i, f'{model.weights_[i]:.2f}', f'{mu[i][0]:.2f}', f'{sqrt(sigma[i][0][0]):.2f}'] for i in range(0, comp)]
    # table = plt.table(cellText=table_vals, colLabels=col_labels, loc='upper right', colWidths=[0.05, 0.1, 0.1, 0.1])
    # table.set_zorder(100) # Artists with lower zorder values are drawn first.

    chi, p = stats.chisquare(counts, f_exp=res)
    print('Значение для n_comp = {} Хи-квадрат = {}, p = {}'.format(comp, chi, p))
    chisquares.append(chi)

    # выделение областей, принадлежащих классу с минимальным математическим ожиданием

    # X = np.expand_dims(field.flatten(), 1)
    # predict = model.predict(X)
    # means = model.means_
    # field_predict = np.resize(predict, field.shape)
    # field_pr_img = field_predict / (comp - 1) * 255
    # field_pr_img = field_pr_img.astype(np.uint8)
    # intensives = [int(i / (comp - 1) * 255) for i in range(0, comp)]

    # получаем маску
    # i = 0
    # for intens in intensives:
    #     value_img = cv.inRange(field_pr_img, intens, intens + 1)
    #     print(model.means_[i])
    #     plt.imshow(value_img, cmap='gray')
    #     plt.show()
    #     i +=

# создание таблицы со значениями хи-квадрат
col_labels = [r'$\it{n}$', r'$\chi^2$']
table_vals = [[comp, f'{chisquares[comp - min(comp_array)]:.2f}'] for comp in comp_array]
table = plt.table(cellText=table_vals, colLabels=col_labels, loc='center right',
                  colWidths=[0.05, 0.2], cellLoc='center')
table.set_zorder(100)  # Artists with lower zorder values are drawn first.
table.set_fontsize(14)
table.scale(1, 2)

plt.grid(True)
plt.legend(fontsize=14)
plt.xlabel(r'Фрактальная размерность, $\it{D}$')
plt.ylabel('Частота')
plt.tight_layout()
plt.show()
print(chisquares)