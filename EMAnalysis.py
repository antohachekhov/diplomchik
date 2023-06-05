import numpy as np
import scipy.stats as stats
from sklearn.mixture import GaussianMixture
from matplotlib import pyplot as plt
from math import sqrt
import cv2 as cv

colors = {'0': '#BEF73E',
          '1': '#C9007A',
          '2': '#FF7F00',
          '3': '#0772A1',
          '4': '#2818B1'}

field = np.loadtxt(r"C:\Users\bortn\Desktop\diplomchik\analysis\new dataset\20\1-04\field_field_prism_w30x1y1.csv", delimiter=';')

chisquares = np.zeros(shape=4)
comp_array = [3]
for comp in comp_array:
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
    counts, bins, _ = plt.hist(field.flatten(), bins=250, facecolor=colors['0'], edgecolor='none', density=True)
    #plt.title('Распределение фрактальных размерностей на изображении')
    plt.title('Кластеризация EM-алгоритмом по ' + str(comp) + ' распределениям')
    for i in range(comp):
        xi = np.linspace(1.5, 3.5, 600)
        #xi = np.linspace(mu[i][0] - 5 * sqrt(sigma[i][0][0]), mu[i][0] + 5 * sqrt(sigma[i][0][0]), 100)
        plt.plot(xi, model.weights_[i] * stats.norm.pdf(xi, mu[i][0], sqrt(sigma[i][0][0])),
                 label=rf'$\mu_{i}={mu[i][0]:.6f}, \sigma_{i}$={sigma[i][0][0]:.6f}', linewidth=2)

    # xi = np.linspace(np.min(bins), np.max(bins), bins.shape[0])
    x = (bins[1:] + bins[:-1]) / 2.
    res = np.zeros(x.shape)
    #for i in range(comp):
    #    res += model.weights_[i] * stats.norm.pdf(x, mu[i][0], sqrt(sigma[i][0][0]))
    #plt.plot(x, res, color='red')
    plt.tight_layout()
    plt.legend()
    plt.grid(True)
    plt.show()

    # chi, p = stats.chisquare(res, f_exp=counts)
    # print('Значение для n_comp = {} Хи-квадрат = {}, p = {}'.format(comp, chi, p))
    # chisquares[j] = chi

    X = np.expand_dims(field.flatten(), 1)
    predict = model.predict(X)
    means = model.means_
    field_predict = np.resize(predict, field.shape)
    field_pr_img = field_predict / (comp - 1) * 255
    field_pr_img = field_pr_img.astype(np.uint8)
    intensives = [int(i / (comp - 1) * 255) for i in range(0, 3)]

    # выделение областей, принадлежащих классу с минимальным математическим ожиданием
    # получаем маску
    i = 0
    for intens in intensives:
        value_img = cv.inRange(field_pr_img, intens, intens + 1)
        print(model.means_[i])
        plt.imshow(value_img, cmap='gray')
        plt.show()
        i += 1
