from PIL import Image as PILImage
from PIL import ImageTk
import numpy as np
import os
from tkinter import *
from tkinter import filedialog
from tkinter.filedialog import askopenfilename
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
import matplotlib.ticker as ticker
import matplotlib.pyplot as plt
import cv2 as cv
from FractalAnalysisClass import FractalAnalysis
import tkinter.messagebox as mb
from sklearn.mixture import GaussianMixture
import scipy.stats as stats
import math
import matplotlib.lines as lns
from functools import partial


class ScaleEdit(Toplevel):
    def __init__(self, parent):
        super().__init__(parent)
        self.title("Расчёта масштаба")
        self.geometry('350x130')

        self.size = StringVar()

        self.top = Frame(self, width=300, height=100)
        self.top.pack(side=TOP)
        self.left = Frame(self.top, width=225, height=100)
        self.left.pack(side=LEFT, pady=20)
        self.right = Frame(self.top, width=75, height=100)
        self.right.pack(side=RIGHT, pady=20, padx=10)
        self.bottom = Frame(self, width=300, height=50)
        self.bottom.pack(side=TOP)

        self.label1 = Label(self.left, text="Введите указанное значение длины", justify=LEFT)
        self.label1.pack(anchor='w')

        self.label2 = Label(self.left, text="линейного масштаба снимка в микрометрах", justify=LEFT)
        self.label2.pack(anchor='w')

        self.input = Entry(self.right, width=10, textvariable=self.size)
        self.input.pack(anchor='w')
        self.input.focus()

        self.button = Button(self.bottom, text="Готово", width=10, command=self.destroy)
        self.button.pack(pady=5)

        self.bind('<Return>', self.close)

    def open(self):
        self.grab_set()
        self.wait_window()
        size = int(self.size.get())
        return size

    def close(self):
        self.destroy()


class Filter(Toplevel):
    def __init__(self, parent, img):
        super().__init__(parent)
        self.title("Фильтрация")
        self.geometry('600x600')

        self.size = StringVar()

        self.bottom_level = StringVar()
        self.top_level = StringVar()

        f = Figure(figsize=(1, 2), dpi=100)
        a = f.add_subplot(111)
        a.hist(img.ravel(), 256, [0, 256])
        a.set_xlabel('Значения уровня серого')
        a.set_ylabel('Количество')

        for label in (a.get_xticklabels() + a.get_yticklabels()):
            label.set_fontsize(7)

        a.xaxis.label.set_size(7)
        a.yaxis.label.set_size(7)
        a.xaxis.set_major_locator(ticker.MultipleLocator(25))

        canvas = FigureCanvasTkAgg(f, self)
        canvas.draw()
        canvas.get_tk_widget().pack(side=BOTTOM, expand=True)

        toolbar = NavigationToolbar2Tk(canvas, self)
        toolbar.update()
        canvas._tkcanvas.pack(side=TOP, fill=BOTH, expand=True)

        self.form = Frame(self, width=400)
        self.form.pack(side=TOP, pady=15)

        self.form_left = Frame(self.form, width=200)
        self.form_left.pack(side=LEFT, padx=30)

        self.label1 = Label(self.form_left, text="Нижний порог", justify=LEFT)
        self.label1.pack(side=LEFT, pady=2)
        self.input1 = Entry(self.form_left, width=10, textvariable=self.bottom_level)
        self.input1.pack(side=LEFT, pady=2)
        self.input1.focus()

        self.form_right = Frame(self.form, width=200)
        self.form_right.pack(side=RIGHT, padx=30)

        self.input2 = Entry(self.form_right, width=10, textvariable=self.top_level)
        self.input2.pack(side=RIGHT, pady=2)
        self.label2 = Label(self.form_right, text="Верхний порог", justify=LEFT)
        self.label2.pack(side=RIGHT, pady=2)

        self.button = Button(self, text="Готово", width=10, command=self.destroy)
        self.button.pack(pady=5)

        self.bind('<Return>', self.close)

        plt.close(f)

    def open(self):
        self.grab_set()
        self.wait_window()
        bottom = int(self.bottom_level.get())
        top = int(self.top_level.get())
        return bottom, top

    def close(self, event):
        self.destroy()


class SaveField(Toplevel):
    def __init__(self, parent):
        super().__init__(parent)

        self.title("Сохранение")
        self.geometry('300x200')

        self.field = parent.field

        self.top = Frame(self)
        self.bottom = Frame(self)
        self.top.pack()
        self.bottom.pack()

        self.save_text = Label(self.top, text="Сохранить поле?")
        self.save_text.pack(pady=20)

        self.button_yes = Button(self.bottom, width=15, text="Да", command=self.save)
        self.button_no = Button(self.bottom, width=15, text="Нет", command=self.destroy)
        self.button_yes.pack(pady=20, padx=30)
        self.button_no.pack(pady=20, padx=30)

    def save(self):
        file_name = filedialog.asksaveasfilename()
        np.savetxt(file_name, self.field, delimiter=";", newline="\n")
        self.destroy()


class EMAnalysisWindow(Toplevel):
    def __init__(self, parent):
        super().__init__(parent)

        self.title("EM-анализ")
        self.geometry('1200x700')

        self.parent = parent
        self.model_out = parent.model
        self.model2 = None
        self.model3 = None
        self.chi1value = 0.
        self.chi2value = 0.

        self.top = Frame(self)
        self.bottom = Frame(self)
        self.top.pack()
        self.bottom.pack()

        self.top_left = Frame(self.top, width=600, height=600)
        self.top_right = Frame(self.top, width=600, height=600)
        self.top_left.pack(side=LEFT)
        self.top_right.pack(side=RIGHT)
        self.top_left.propagate(FALSE)
        self.top_right.propagate(FALSE)

        self.bottom_left = Frame(self.bottom, width=600, height=100)
        self.bottom_right = Frame(self.bottom, width=600, height=100)
        self.bottom_left.pack(side=LEFT)
        self.bottom_right.pack(side=RIGHT)
        self.bottom_left.propagate(FALSE)
        self.bottom_right.propagate(FALSE)

        self.line_right = Frame(self.bottom_right)
        self.line_right.pack(pady=5)

        self.chi1 = StringVar()
        self.chi1.set("")
        self.chi2 = StringVar()
        self.chi2.set("")
        self.chi1_label = Label(self.bottom_left, textvariable=self.chi1)
        self.chi2_label = Label(self.line_right, textvariable=self.chi2)
        self.chi1_label.pack(pady=5)
        self.chi2_label.pack(side=LEFT, padx=40)

        self.button1 = Button(self.bottom_left, width=30, height=2, text="Первый метод", command=self.comp2)
        self.button2 = Button(self.bottom_right, width=30, height=2, text="Второй метод", command=self.comp3)
        self.button1.pack(pady=10, padx=70, side=RIGHT)
        self.button2.pack(pady=10, padx=70, side=RIGHT)

        # self.EM(parent, 2, self.top_left, self.chi1)
        # self.EM(parent, 3, self.top_right, self.chi2)

        self.f1 = Figure(figsize=(1, 1), dpi=100)
        self.a1 = self.f1.add_subplot(111)
        self.canvas1 = FigureCanvasTkAgg(self.f1, self.top_left)
        self.canvas1.get_tk_widget().pack(side=BOTTOM, expand=True)
        self.toolbar1 = NavigationToolbar2Tk(self.canvas1, self.top_left)
        self.canvas1.tkcanvas.pack(side=TOP, fill=BOTH, expand=True)

        self.f2 = Figure(figsize=(1, 1), dpi=100)
        self.a2 = self.f2.add_subplot(111)
        self.canvas2 = FigureCanvasTkAgg(self.f2, self.top_right)
        self.canvas2.get_tk_widget().pack(side=BOTTOM, expand=True)
        self.toolbar2 = NavigationToolbar2Tk(self.canvas2, self.top_right)
        self.canvas2.tkcanvas.pack(side=TOP, fill=BOTH, expand=True)

        self.EM2 = partial(self.EM, parent.field, 2, self.chi1, self.a1, self.canvas1, self.toolbar1)
        self.EM3 = partial(self.EM, parent.field, 3, self.chi2, self.a2, self.canvas2, self.toolbar2)

        self.button_change1 = Button(self.bottom_left, width=30, height=2, text="Повторить", command=self.EM2)
        self.button_change2 = Button(self.bottom_right, width=30, height=2, text="Повторить", command=self.EM3)
        self.button_change1.pack(pady=5, padx=20, side=LEFT)
        self.button_change2.pack(pady=5, padx=20, side=LEFT)

        set_model_in_parent = partial(self.set_model, parent)

        self.check_var = BooleanVar()
        self.check_var.set(FALSE)
        self.check_button = Checkbutton(self.line_right, text='Использовать эту модель', variable=self.check_var,
                                        onvalue=True, offvalue=False, command=set_model_in_parent)
        self.check_button.pack(side=RIGHT, padx=40)

        self.chi1value = self.EM2()
        self.chi2value = self.EM3()

    def get_comp(self):
        self.grab_set()
        self.wait_window()
        if self.chi2value > self.chi2value:
            getcomp = 2
        else:
            getcomp = 3
        return getcomp

    def set_model(self, parent):
        parent.model = self.model3

    def EM(self, field, comp, chi_text, a, canvas, toolbar):
        a.clear()

        a.set_title('Кластеризация EM-алгоритмом по ' + str(comp) + ' распределениям')
        a.set_xlabel('Фрактальная размерность')
        a.set_ylabel('Количество')

        a.grid(True)
        for label in (a.get_xticklabels() + a.get_yticklabels()):
            label.set_fontsize(7)
        a.xaxis.label.set_size(7)
        a.yaxis.label.set_size(7)

        X = np.expand_dims(field.flatten(), 1)
        model = GaussianMixture(n_components=comp, covariance_type='full')
        model.fit(X)
        mu = model.means_
        sigma = model.covariances_
        if comp == 3:
            self.model3 = model
        # вывод гистограммы распределений фрактальных размерностей в поле
        # count - кол-во, bins - значение размерности
        counts, bins, _ = a.hist(field.flatten(), bins=250, facecolor="#BEF73E", edgecolor='none',
                                 density=True)
        x = (bins[1:] + bins[:-1]) / 2.
        res = np.zeros(x.shape)
        for i in range(comp):
            res += model.weights_[i] * stats.norm.pdf(x, mu[i][0], math.sqrt(sigma[i][0][0]))

        a.plot(x, res, color='red')
        canvas.draw()
        toolbar.update()

        chi, p = stats.chisquare(counts, f_exp=res)
        chi_text.set("Хи-квадрат = {}, p = {}".format(chi, p))

        return chi

    def comp2(self):
        self.parent.comp = 2
        self.destroy()

    def comp3(self):
        self.parent.comp = 3
        self.destroy()


class Parameters(Toplevel):
    def __init__(self, parent):
        super().__init__(parent)

        self.title("Установить параметры")
        self.geometry('250x220')

        self.parent = parent

        self.minsamples = IntVar()
        self.minsamples.set(8)
        self.min_transition = DoubleVar()
        self.min_transition.set(0.4)
        self.median_kernel = IntVar()
        self.median_kernel.set(21)

        self.form = Frame(self, width=250, height=70)
        self.form.pack()
        self.form.propagate(FALSE)

        self.minsamples_label = Label(self.form, text="min_samples")
        self.minsamples_label.grid(row=0, column=0, padx=20, pady=10, sticky=W)
        self.minsamples_input = Entry(self.form, textvariable=self.minsamples, width=10)
        self.minsamples_input.grid(row=0, column=1, pady=10, sticky=E)

        self.mintransition_label = Label(self.form, text="min_transition")
        self.mintransition_label.grid(row=1, column=0, padx=20, pady=10, sticky=W)
        self.mintransition_input = Entry(self.form, textvariable=self.min_transition, width=10)
        self.mintransition_input.grid(row=1, column=1, pady=10, sticky=E)

        self.mediankernel_label = Label(self.form, text="median_kernel")
        self.mediankernel_label.grid(row=2, column=0, padx=20, pady=10, sticky=W)
        self.mediankernel_input = Entry(self.form, textvariable=self.median_kernel, width=10)
        self.mediankernel_input.grid(row=2, column=1, pady=10, sticky=E)

        self.set_button = Button(self, width=20, height=1, text="Установить параметры", command=self.set)
        self.set_button.pack(pady=2)

        self.flag_set = StringVar()
        self.flag_set.set("")
        self.flag_label = Label(self, textvariable=self.flag_set)
        self.flag_label.pack(pady=2)

        self.close_button = Button(self, width=20, height=1, text="Закрыть", command=self.destroy)
        self.close_button.pack(pady=2)

    def set(self):
        self.parent.param[0] = self.minsamples.get()
        self.parent.param[1] = self.min_transition.get()
        self.parent.param[2] = self.median_kernel.get()
        self.flag_set.set("Параметры установлены")


class App(Tk):
    def __init__(self):
        super().__init__()  # Создаём окно приложения
        self.title("Входные данные")
        self.geometry('500x400')

        # Входные данные
        self.ScaleCoef = 0.  # коэффициент масштабирования метр/пиксель
        self.img = None
        self.field = None
        self.winsize = IntVar()
        self.dx = IntVar()
        self.dy = IntVar()
        self.fractal = FractalAnalysis(show_step=False)
        self.comp = 0
        self.param = [0, 0., 0]
        self.model = None

        # Данные для отображения
        self.Photo = None
        self.InputImage = None

        self.f_top = Frame(self)
        self.f_bottom = Frame(self)
        self.f_top.pack()
        self.f_bottom.pack()

        self.area_button = Frame(self.f_bottom, width=500, height=75)
        self.area_button.pack()

        self.frame_photo = LabelFrame(self.f_top, text="Изображение", width=220, height=320)
        self.top_right = Frame(self.f_top, width=265, height=320)
        self.frame_field = LabelFrame(self.top_right, text="Поле", width=265, height=320)
        self.frame_photo.pack(side=LEFT, padx=5, pady=5)
        self.top_right.pack(side=RIGHT, padx=5, pady=5)
        self.frame_field.pack(side=TOP, pady=5)

        self.scale_text = StringVar()
        self.scale_text.set('Масштаб не измерен')
        self.filter_result_text = StringVar()
        self.filter_result_text.set("")

        self.canvas_photo = Canvas(self.frame_photo, height=150, width=200, background='gray')
        self.scale_label = Label(self.frame_photo, textvariable=self.scale_text)
        self.button_load_photo = Button(self.frame_photo, height=1, width=25, text="Загрузить изображение",
                                        command=self.input_image)
        self.button_filter_windows = Button(self.frame_photo, height=1, width=25, text="Применить фильтрацию",
                                            command=self.filter_image)
        self.result_filter = Label(self.frame_photo, textvariable=self.filter_result_text)
        self.canvas_photo.pack(padx=10, pady=10)
        self.scale_label.pack(pady=3)
        self.button_load_photo.pack(pady=7)
        self.button_filter_windows.pack(pady=7)
        self.result_filter.pack(pady=3)

        self.field_check = BooleanVar()
        self.flag_field_check = Checkbutton(self.frame_field, text='Вычислить поле', variable=self.field_check,
                                            onvalue=True, offvalue=False, command=self.switchButtonState)
        self.flag_field_check.pack(pady=3, side=TOP, anchor=NW)
        self.input_field_button = Button(self.frame_field, height=1, width=25, text="Загрузить поле",
                                         command=self.input_field)
        self.input_field_button.pack(pady=3, padx=30)
        self.input_field_text = StringVar()
        self.input_field_text.set("")
        self.input_field_result = Label(self.frame_field, textvariable=self.input_field_text)
        self.input_field_result.pack(pady=3)
        self.show_field_button = Button(self.frame_field, height=1, width=25, text="Показать поле",
                                        command=self.field_to_image)
        self.show_field_button.pack(pady=3, padx=30)
        self.show_field_button['state'] = DISABLED

        self.field_param_label = Label(self.frame_field, text="Параметры оконной обработки", font="Helvetica 8 bold")
        self.field_param_label.pack(pady=3)

        self.field_param_line1 = Frame(self.frame_field, width=265)
        self.field_param_line2 = Frame(self.frame_field, width=265)
        self.field_param_line3 = Frame(self.frame_field, width=265)

        self.filed_param_winsize = Label(self.field_param_line1, width=20, text="Размер окна", justify=LEFT)
        self.filed_param_winsize_input = Entry(self.field_param_line1, width=10, textvariable=self.winsize)
        self.filed_param_dx = Label(self.field_param_line2, width=20, text="Шаг по x", justify=LEFT)
        self.filed_param_dx_input = Entry(self.field_param_line2, width=10, textvariable=self.dx)
        self.filed_param_dy = Label(self.field_param_line3, width=20, text="Шаг по y", justify=LEFT)
        self.filed_param_dy_input = Entry(self.field_param_line3, width=10, textvariable=self.dy)

        self.field_param_line1.pack(pady=3)
        self.field_param_line2.pack(pady=3)
        self.field_param_line3.pack(pady=3)
        self.filed_param_winsize.pack(side=LEFT)
        self.filed_param_winsize_input.pack(side=RIGHT)
        self.filed_param_dx.pack(side=LEFT)
        self.filed_param_dx_input.pack(side=RIGHT)
        self.filed_param_dy.pack(side=LEFT)
        self.filed_param_dy_input.pack(side=RIGHT)

        self.start_analysis_button = Button(self.area_button, text="Начать анализ", width=25,
                                            command=self.start_analysis)
        self.start_analysis_button.pack(pady=10)

        self.setparam_button = Button(self.top_right, text="Установить параметры", width=20, height=1,
                                      command=self.setParam)
        self.setparam_button.pack(pady=5)
        self.clear_button = Button(self.top_right, text="Очистить данные", width=20, height=1, command=self.clear)
        self.clear_button.pack(pady=5)

        self.protocol("WM_DELETE_WINDOW", self.on_closing)

    def on_closing(self):
        plt.close()
        self.destroy()

    def setParam(self):
        param_win = Parameters(self)
        param_win.wait_window()
        self.fractal.set_param(self.param[0], self.param[1], self.param[2])

    def clear(self):
        self.ScaleCoef = 0.  # коэффициент масштабирования метр/пиксель
        self.img = None
        self.field = None
        self.winsize.set(0)
        self.dx.set(0)
        self.dy.set(0)
        self.comp = 0
        self.Photo = None
        self.InputImage = None
        self.scale_text.set('Масштаб не измерен')
        self.input_field_text.set("")
        self.field_check.set(FALSE)
        self.input_field_button['state'] = NORMAL
        self.model = None
        self.filter_result_text.set("")

    def start_analysis(self):
        if self.img is not None and self.winsize.get() != 0 and self.dx.get() != 0 and self.dy.get() != 0:
            if self.field is None:
                self.fractal.set_field(self.img, self.winsize.get(), self.dx.get(), self.dy.get(), method='prism')
                self.field = self.fractal.field
                self.show_field_button['state'] = NORMAL
                save = SaveField(self)
                save.wait_window()
            else:
                self.fractal.set_field(self.img, self.winsize.get(), self.dx.get(), self.dy.get(), field=self.field)
            # em = EMAnalysisWindow(self)
            # self.comp = em.get_comp()
            self.EMcompare()
            print('Выбран метод для {} распределений'.format(self.comp))
            if self.comp in [2, 3]:
                if self.model is not None and self.comp == 3:
                    self.fractal.set_EMmodel(self.model)
                mask_stratum = self.fractal.segment_stratum(self.comp)

                # plt.imshow(cv.cvtColor(mask_stratum, cv.COLOR_GRAY2RGB))
                # plt.show()

                img = cv.cvtColor(self.img, cv.COLOR_BGR2RGB)
                # создание слоя-заливки
                color_layer = np.zeros(img.shape, dtype=np.uint8)
                color_layer[:] = (30, 21, 117) if self.comp == 2 else (255, 0, 102)

                # наложение заливки на изображение через маску
                # получаются закрашенные сегменты переходного слоя
                stratum = cv.bitwise_and(img, color_layer, mask=mask_stratum)

                # соединение исходного изображения с полупрозрачными закрашенными сегментами
                result = cv.addWeighted(img, 1, stratum, 0.2, 0.0)

                # выделение контуров сегментов
                color_contours = (0, 0, 255) if self.comp == 2 else (102, 0, 153)
                contours, hierarchy = cv.findContours(mask_stratum, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
                cv.drawContours(result, contours, -1, color_contours, 1, cv.LINE_AA, hierarchy, 3)

                # перевод результата в пространство RGB
                result = cv.cvtColor(result, cv.COLOR_BGR2RGB)

                color_square = (1, 0, 0) if self.comp == 2 else (0.4, 0, 0.6)
                plt.figure(figsize=(13, 7))
                plt.imshow(result)
                purple_square = lns.Line2D([], [], color=color_square, marker='s', linestyle='None',
                                           markersize=7, label='Переходный слой')
                plt.legend(handles=[purple_square])

                if self.comp == 2:
                    distances, coor = self.fractal.distances_curves
                    distances2, coor2 = self.fractal.distances_curves_vertical
                    mean_d_1 = np.mean(distances)
                    mean_d_2 = np.mean(distances2)
                    print("Средняя ширина переходного слоя по первому методу: {:e}".format(mean_d_1 * self.ScaleCoef))
                    print("Средняя ширина переходного слоя по второму методу: {:e}".format(mean_d_2 * self.ScaleCoef))
                    for i, d in enumerate(distances):
                        plt.text(coor[i, 0, 0], coor[i, 0, 1], "{:e}".format(d * self.ScaleCoef),
                                 bbox=dict(boxstyle="round,pad=0.3", fc="#d69f67", ec="black", lw=1))
                    for i, d in enumerate(distances2):
                        plt.text(coor2[i, 1, 0], coor2[i, 1, 1], "{:e}".format(d * self.ScaleCoef),
                                 bbox=dict(boxstyle="round,pad=0.3", fc="#674D92", ec="white", lw=1))

                plt.show()
        else:
            mb.showerror("Ошибка", 'Введены не все данные!')

    def EMcompare(self):
        chi1 = 0.
        chi2 = 0.
        for comp in [2, 3]:
            field_array = np.expand_dims(self.field.flatten(), 1)
            model = GaussianMixture(n_components=comp, covariance_type='full')
            model.fit(field_array)
            mu = model.means_
            sigma = model.covariances_
            if comp == 3:
                self.model = model
            # вывод гистограммы распределений фрактальных размерностей в поле
            # count - кол-во, bins - значение размерности
            counts, bins, _ = plt.hist(self.field.flatten(), bins=250, facecolor="#BEF73E", edgecolor='none',
                                       density=True)
            x = (bins[1:] + bins[:-1]) / 2.
            res = np.zeros(x.shape)
            for i in range(comp):
                res += model.weights_[i] * stats.norm.pdf(x, mu[i][0], math.sqrt(sigma[i][0][0]))

            chi, p = stats.chisquare(counts, f_exp=res)
            plt.close()
            if comp == 2:
                chi1 = chi
            else:
                chi2 = chi
        if chi1 < chi2:
            self.comp = 2
        else:
            self.comp = 3

    def switchButtonState(self):
        if self.field_check.get():
            self.input_field_button['state'] = DISABLED
            self.filed_param_winsize_input.focus()
        else:
            self.input_field_button['state'] = NORMAL

    def input_field(self):
        Test = False
        if Test:
            print('Error')
            uploadfilename = None
            # uploadfilename = tests[nametest]['field']
            # win_size = tests[nametest]['win_size']
            # dx = tests[nametest]['dx']
            # dy = tests[nametest]['dy']
        else:
            # чтение фрактального поля из файла
            Tk().withdraw()
            uploadfilename = askopenfilename()

        self.field = np.loadtxt(uploadfilename, delimiter=";")
        self.input_field_text.set("Поле загружено")
        self.show_field_button['state'] = NORMAL
        self.filed_param_winsize_input.focus()

    def field_to_image(self, show=True):
        """
        Перевод поля фрактальных размерностей в пространство оттенков серого

        :param input_field: 2-D numpy массив
            Поле фрактальных размерностей.
        :param show: bool
            True, если необходимо вывести изображение визуализации поля

        :return img_out: 2-D numpy массив
            Изображение-визуализация поля фрактальных размерностей
        """
        img_out = np.empty((self.field.shape[0], self.field.shape[1]), dtype=np.uint8)
        for i in range(img_out.shape[0]):
            for j in range(img_out.shape[1]):

                if self.field[i][j] - 1.0 > 0.0:
                    img_out[i][j] = int(255 * (self.field[i][j] - 1) / 3)
                else:
                    img_out[i][j] = 0
        if show:
            plt.imshow(img_out, cmap='gray')
            plt.show()
        return img_out

    def input_image(self):
        image_uploaded = False
        while not image_uploaded:
            Test = False
            # получение имени файла и пути к нему
            if Test:
                # image_name = tests[nametest]['image']
                print('Error')
                image_name = '.'
            else:
                Tk().withdraw()
                image_name = askopenfilename()
            path_with_name, file_extension = os.path.splitext(image_name)
            folder, filename = os.path.split(path_with_name)

            # чтение изображения из файла
            try:
                # читаем изображение в оттенках серого
                self.InputImage = PILImage.open(image_name).convert('L')
                img_original = np.asarray(self.InputImage)

                exifScale = False
                if file_extension == '.tif' or file_extension == '.TIF':
                    exifScale = True
                    exif_data = PILImage.open(image_name).getexif()
                    try:
                        tags = exif_data.items()._mapping[34118]

                        width = 0.
                        measurement_dict = {
                            "m": 1E-3,
                            "µ": 1E-6,
                            "n": 1E-9,
                        }

                        for row in tags.split("\r\n"):
                            key_value = row.split(" = ")
                            if len(key_value) == 2:
                                if key_value[0] == "Width":
                                    width = key_value[1]
                                    break
                        width_str = width.split(" ")

                        width_unit = width_str[1]
                        width_value = float(width_str[0]) * measurement_dict[width_unit[0]]
                        self.ScaleCoef = width_value / float(exif_data.items()._mapping[256])
                    except Exception:
                        exifScale = False
                if not exifScale:
                    n = 0
                    while img_original[700, 21 + n + 1] > 240:
                        n += 1
                    n += 5

                    # print('Enter the size of the scale segment in micrometer')

                    input_scale_window = ScaleEdit(self)
                    size = input_scale_window.open()

                    size_segment = float(size) * 1e-6
                    self.ScaleCoef = size_segment / n

                img = np.asarray(self.InputImage.crop((0, 0, self.InputImage.size[0], self.InputImage.size[1] - 103)))

            except Exception as e:
                # вызывается исключение, если на вход был подан не поддерживаемый формат изображения
                mb.showerror("Error", e)
                continue

            image_uploaded = True

        self.Photo = ImageTk.PhotoImage(self.InputImage.resize((self.canvas_photo.winfo_width(),
                                                                self.canvas_photo.winfo_height())))
        self.canvas_photo.create_image(0, 0, anchor='nw', image=self.Photo)

        self.scale_text.set('Коэф. масштабирования = {:.3e}'.format(self.ScaleCoef))

        self.img = img

    def filter_image(self):
        filter_window = Filter(self, self.img)
        bottom, top = filter_window.open()
        self.img = self.noise(bottom, top, plot=True)
        self.filter_result_text.set("Изображение профильтровано")

    def noise(self, lower, upper, plot=False):
        """
        Функция избавляется от шумов на изображении (чёрные или белые пятна)

        :param lower: Int
            Нижний предел допустимой (не шумовой) интенсивности цвета.
        :param upper: Int
            Верхний предел допустимой (не шумовой) интенсивности цвета.
        :param plot: Bool.
            True, если необходимо вывести пошаговую обработку изображения.
            False, если это не нужно.

        :return: 3-D numpy массив размерности NxMx3.
            Обработанное изображение в цветовом пространстве BGR.
        """
        # перевод изображения в пространство оттенков серого
        # gray = cv.cvtColor(img_orig, cv.COLOR_BGR2GRAY)

        # Пороговая бинаризация изображение
        # пиксели, интенсивность которых не входит в диапазон [lower, upper], становятся чёрными. Остальные - белыми
        mask1 = cv.threshold(self.img, lower, 255, cv.THRESH_BINARY_INV)[1]
        mask2 = cv.threshold(self.img, upper, 255, cv.THRESH_BINARY)[1]
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
        radius = int(self.img.shape[0] / 3) + 1
        if radius % 2 == 0:
            radius = radius + 1

        # Размываем изображение
        median = cv.medianBlur(self.img, radius)

        # Накладываем маску на изображение
        img_masked = cv.bitwise_and(self.img, mask)

        # Накладываем инвертированную маску на размытое изображение
        median_masked = cv.bitwise_and(median, mask_inv)

        # Соединяем два полученных изображения
        result = cv.add(img_masked, median_masked)

        if plot:
            plt.subplot(2, 3, 1)
            plt.imshow(self.img, cmap='gray')
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


app = App()
app.mainloop()

"""
def _count(a, axis=None):
    if hasattr(a, 'count'):
        num = a.count(axis=axis)
        if isinstance(num, np.ndarray) and num.ndim == 0:
            # In some cases, the `count` method returns a scalar array (e.g.
            # np.array(3)), but we want a plain integer.
            num = int(num)
    else:
        if axis is None:
            num = a.size
        else:
            num = a.shape[axis]
    return num

def power_divergence(f_obs, f_exp=None, ddof=0, axis=0, lambda_=1):
    f_obs = np.asanyarray(f_obs)

    if f_exp is not None:
        f_exp = np.asanyarray(f_exp)
    else:
        # Ignore 'invalid' errors so the edge case of a data set with length 0
        # is handled without spurious warnings.
        with np.errstate(invalid='ignore'):
            f_exp = f_obs.mean(axis=axis, keepdims=True)

    # `terms` is the array of terms that are summed along `axis` to create
    # the test statistic.  We use some specialized code for a few special
    # cases of lambda_.
    if lambda_ == 1:
        # Pearson's chi-squared statistic
        terms = (f_obs.astype(np.float64) - f_exp)**2 / f_exp
    elif lambda_ == 0:
        # Log-likelihood ratio (i.e. G-test)
        terms = 2.0 * special.xlogy(f_obs, f_obs / f_exp)
    elif lambda_ == -1:
        # Modified log-likelihood ratio
        terms = 2.0 * special.xlogy(f_exp, f_exp / f_obs)
    else:
        # General Cressie-Read power divergence.
        terms = f_obs * ((f_obs / f_exp)**lambda_ - 1)
        terms /= 0.5 * lambda_ * (lambda_ + 1)

    stat = terms.sum(axis=axis)

    num_obs = _count(terms, axis=axis)
    ddof = np.asarray(ddof)
    p = stats.distributions.chi2.sf(stat, num_obs - 1 - ddof)

    return stat, p

def chi2(a1, a2):
    sum = np.sum(np.power(a1 / a1.shape[0] - a2 / a2.shape[0], 2) / (a1 + a2))
    chi = a1.shape[0] * a2.shape[0] * sum
    # sum = np.sum(np.power(a1 - a2, 2) / (a1 + a2))
    # chi = sum
    return chi

comp = 3
for comp in [2, 3]:
    field = np.loadtxt(r"C:\\Users\bortn\Desktop\diplomchik\analysis\old dataset\20min_1\20min1_dx1dy1w30.csv",
                       delimiter=";")
    X = np.expand_dims(field.flatten(), 1)
    model = GaussianMixture(n_components=comp, covariance_type='full')
    model.fit(X)
    mu = model.means_
    sigma = model.covariances_

    counts, bins, o = plt.hist(field.flatten(), bins=50, facecolor="#BEF73E", edgecolor='none', density=True)
    # counts = np.loadtxt(r"C:\\Users\bortn\Desktop\diplomchik\analysis\new dataset\5\1-11\counts.csv", delimiter=";")
    # bins = np.loadtxt(r"C:\\Users\bortn\Desktop\diplomchik\analysis\new dataset\5\1-11\bins.csv", delimiter=";")
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

    np.savetxt(r"C:\\Users\bortn\Desktop\diplomchik\analysis\old dataset\20min_1\res{}.csv".format(comp), res, delimiter=";")
    np.savetxt(r"C:\\Users\bortn\Desktop\diplomchik\analysis\old dataset\20min_1\count{}.csv".format(comp), counts,
               delimiter=";")
    plt.plot(x, res, color='red', label='res')
    plt.plot(x, counts, color='green', label='counts')
    plt.tight_layout()
    plt.legend()
    plt.grid(True)
    plt.show()

    chi, p = stats.chisquare(res, f_exp=counts)
    chisq = chi2(res, counts)
    print('Значение для n_comp = {} Хи-квадрат = {}, p = {}'.format(comp, chi, p))
    print('Значение для n_comp = {} Хи-квадрат = {}'.format(comp, chisq))



np.random.seed(58008)


def normalise(func):
    o = []
    for p in func:
        res = median(p)
        res = easy_mean(res)
        o.append(res)
    return o


def noised(func, k=0.3, fitob=0.03):
    o = []
    for p in func:
        r = (np.random.random()*2-1) * k

        # Standard noise and random emissions
        if np.random.random() < fitob: c = p + r*7
        else: c = p + r

        o.append(c)
    return o


def arith_mean(f, buffer_size=10):
    # Creating buffer
    if not hasattr(arith_mean, "buffer"):
        arith_mean.buffer = [f] * buffer_size

    # Move buffer to actually values ( [0, 1, 2, 3] -> [1, 2, 3, 4] )
    arith_mean.buffer = arith_mean.buffer[1:]
    arith_mean.buffer.append(f)

    # Calculation arithmetic mean
    mean = 0
    for e in arith_mean.buffer: mean += e
    mean /= len(arith_mean.buffer)

    return mean


def easy_mean(f, s_k=0.2, max_k=0.8, d=0.5):
    # Creating static variable
    if not hasattr(easy_mean, "fit"):
        easy_mean.fit = f

    # Adaptive ratio
    k = s_k if (abs(f - easy_mean.fit) < d) else max_k

    # Calculation easy mean
    easy_mean.fit += (f - easy_mean.fit) * k

    return easy_mean.fit


def median(f):
    # Creating buffer
    if not hasattr(median, "buffer"):
        median.buffer = [f] * 3

    # Move buffer to actually values ( [0, 1, 2] -> [1, 2, 3] )
    median.buffer = median.buffer[1:]
    median.buffer.append(f)

    # Calculation median
    a = median.buffer[0]
    b = median.buffer[1]
    c = median.buffer[2]
    middle = max(a, c) if (max(a, b) == max(b, c)) else max(b, min(a, c))

    return middle


def kalman(f, q=0.25, r=0.7):
    if not hasattr(kalman, "Accumulated_Error"):
        kalman.Accumulated_Error = 1
        kalman.kalman_adc_old = 0

    if abs(f-kalman.kalman_adc_old)/50 > 0.25:
        Old_Input = f*0.382 + kalman.kalman_adc_old*0.618
    else:
        Old_Input = kalman.kalman_adc_old

    Old_Error_All = (kalman.Accumulated_Error**2 + q**2)**(1/2)
    H = Old_Error_All**2/(Old_Error_All**2 + r**2)
    kalman_adc = Old_Input + H * (f - Old_Input)
    kalman.Accumulated_Error = ((1 - H)*Old_Error_All**2)**(1/2)
    kalman.kalman_adc_old = kalman_adc

    return kalman_adc

column = np.loadtxt(r"C:\\Users\bortn\Desktop\diplomchik\analysis\new dataset\5\1-11\column.csv", delimiter=",")
median_column = medfilt(column, kernel_size=11)

o = []
for p in column:
    res = easy_mean(p)
    o.append(res)
o_array = np.array(o)

merge = []
for p in median_column:
    res = easy_mean(p)
    merge.append(res)
merge_array = np.array(merge)

# merge = normalise(column)

fig, ax = plt.subplots()

ax.plot(column, label='orig')
ax.plot(median_column, label='median')
ax.plot(o_array, label='adaptive')
ax.plot(merge_array, label='median+adaptive')
ax.legend()
plt.show()


def easy_mean(f, s_k=0.2, max_k=0.9, d=0.4):
    # Creating static variable
    if not hasattr(easy_mean, "fit"):
        easy_mean.fit = f

    # Adaptive ratio
    k = s_k if (abs(f - easy_mean.fit) < d) else max_k

    # Calculation easy mean
    easy_mean.fit += (f - easy_mean.fit) * k

    return easy_mean.fit




# вспомогательные флаги-переменные
# flag_convert = False
# flag_filter = False

# field_uploaded = False

# coef_scale = None  # коэффициент масштабирования метр/пиксель






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

# вывод загруженного/обработанного изображения
plt.imshow(img, cmap='gray')

if flag_filter:
    plt.title("Input filtered image")
else:
    plt.title('Input image')
plt.show()






measurement_dict = {
    "m": 1E-3,
    "µ": 1E-6,
    "n": 1E-9,
}

img_file = Image.open(r"")
exif_data = img_file.getexif()

tags = exif_data.items()._mapping[34118]

width = 0.

for row in tags.split("\r\n"):
    key_value = row.split(" = ")
    if len(key_value) == 2:
        if key_value[0] == "Width":
            width = key_value[1]
            break

print(width)

width_str = width.split(" ")

width_unit = width_str[1]
width_value = float(width_str[0]) * measurement_dict[width_unit[0]]


print(width_value, 'm')

print('Done')




def f_kas(fx0, dfx0, x0, x):
    kas = x.copy()
    kas -= x0
    kas *= dfx0
    kas += fx0
    return kas


def f_norm(fx0, dfx0, x0, x):
    norm = x.copy()
    norm -= x0
    norm *= 1. / (dfx0)
    norm = fx0 - norm
    return norm


class Line:

    def __init__(self, a=1., b=1., c=0.):
        self.a = a
        self.b = b
        self.c = c

    def y(self, x):
        return (self.a / self.b) * x + (self.c / self.b)

    def general_equation(self, points):
        if points.ndim == 1:
            if points.shape[0] != 2:
                raise ValueError('Number of features must be 2')
            else:
                return self.a * points[0] - self.b * points[1] + self.c
        elif points.ndim == 2:
            if points.shape[1] != 2:
                raise ValueError('Number of features must be 2')
            else:
                return self.a * points[:, 0] - self.b * points[:, 1] + self.c
        else:
            raise ValueError('Points have incorrect dimension')


def intersection_line_curve(k_line, z_line, curve, eps=1e-7):
    if curve.shape[1] != 2:
        raise ValueError('Number of features must be 2')

    if k_line > 0:
        # Right
        sign_direction = 0
        curve_direction = -1

    else:  # k_line < 0
        # Left
        sign_direction = -1
        curve_direction = 1

    lineModel = Line(a=k_line, c=z_line)
    eq = lineModel.general_equation(curve)
    eq[np.abs(eq) < eps] = 0.
    eq_sign = np.sign(eq)

    if np.all(eq_sign == eq_sign[0]):
        raise ValueError('Line and a curve do not intersect')

    eq_signchange = ((np.roll(eq_sign, curve_direction) - eq_sign) != 0).astype(int)
    change_index = np.where(eq_signchange == 1)[0]

    if eq_sign[change_index[sign_direction]] == 0:
        p = curve[change_index[sign_direction]]
    else:
        a = curve[change_index[sign_direction] - curve_direction]
        b = curve[change_index[sign_direction]]

        k_segment = (b[1] - a[1]) / (b[0] - a[0])
        z_segment = (b[1] - a[1]) * (-a[0] / (b[0] - a[0]) + a[1] / (b[1] - a[1]))

        p = np.array([(z_segment - z_line) / (k_line - k_segment),
                      (k_line * z_segment - k_segment * z_line) / (k_line - k_segment)])

    return p



def functions_distance(curve1, curve2, eps=1e-7, n_diff=3):
    if curve1.shape[0] < 2 or curve2.shape[0] < 2:
        raise ValueError('Number of function points must be at least 2')
    if curve1.shape[1] != 2 or curve2.shape[1] != 2:
        raise ValueError('Number of features must be 2')

    hx = np.mean(np.diff(curve1[:, 0]))
    distance = 0.
    n = 0
    for (index,), x0 in np.ndenumerate(curve1[:, 0]):
        if index < n_diff:
            [dx, dy] = curve1[index + n_diff] - curve1[index]
        elif index >= curve1.shape[0] - n_diff:
            [dx, dy] = curve1[index] - curve1[index - n_diff]
        else:
            [dx, dy] = curve1[index + n_diff] - curve1[index - n_diff]

        if math.fabs(dy) < eps:
            print("In x = {} the normal is directed strictly vertically".format(x0))
            difarray = np.abs(curve1[:, 0] - x0)
            index = np.argmin(difarray)
            if difarray[index] <= hx:
                p = curve1[index]
                distance += math.dist(curve1[index], p)
                n += 1

                continue
        if math.fabs(dx) < eps:
            print("In x = {} the normal is directed strictly horizontally".format(x0))
            continue

        k_norm = -1. / (dy / dx)
        z_norm = curve1[index, 1] + x0 / (dy / dx)

        if k_norm > 0:
            # Right
            x_min = x0
            x_max = curve2[-1, 0]

        else:  # k_line < 0
            # Left
            x_min = curve2[0, 0]
            x_max = x0

        curve2range = curve2[np.logical_and(x_min <= x, x_max >= x)]

        try:
            p = intersection_line_curve(k_norm, z_norm, curve2range)
        except Exception:
            continue
        distance += math.dist(curve1[index], p)
        n += 1

    distance /= n
    return distance


# if index % 10 == 0:
#     plt.plot(curve1[:, 0], curve1[:, 1])
#     plt.plot(curve2[:, 0], curve2[:, 1])
#     plt.plot([curve1[index, 0], p[0]], [curve1[index, 1], p[1]])
#     plt.scatter(p[0], p[1])
#     plt.scatter(curve1[index, 0], curve1[index, 1])
#     plt.show()



# x = -1.4181818
# x0 = -6
# normal = Line(a=-1./-1.2, c=3.6+x0/-1.2)
# print(normal.y(x0))

def myfunc(x):
    y = x**2
    return y


error = False
points = False
inclination = False


eps = 0.0000001
h = 0.1
x = np.arange(-7, 7 + h, h)
f = np.vectorize(math.sin)
# f = np.vectorize(myfunc)
y1 = f(x)
f2 = np.vectorize(math.sin)
y2 = f2(x) + 5
y1array = np.array([x, y1]).T
y2array = np.array([x, y2]).T

fig, ax = plt.subplots(1)
ax.plot(y1array[:, 0], y1array[:, 1])
ax.plot(y2array[:, 0], y2array[:, 1])
ax.xaxis.set_major_locator(ticker.MultipleLocator(0.5))
ax.xaxis.set_minor_locator(ticker.MultipleLocator(0.1))
ax.yaxis.set_major_locator(ticker.MultipleLocator(0.5))
ax.yaxis.set_minor_locator(ticker.MultipleLocator(0.1))
ax.grid(which='major')
ax.grid(which='minor')
plt.show()

ind = 80
x0 = x[ind]
n = 3

d = functions_distance(y1array, y2array)
print(d)


dy = y1[ind + n] - y1[ind - n]
if math.fabs(dy) < eps:
    # dy = 0.
    # нормаль направленна вверх
    difarray = np.abs(y2array[:, 0] - x0)
    index = np.argmin(difarray)
    if difarray[index] <= 2 * h:
        p = y2array[index]
        print(p)
else:

    inclination = True
    left = False

    dx = h * n * 2
    dfx0 = dy / dx
    fx0 = y1[ind]
    # Method 1
    k_norm = -1. / dfx0
    z_norm = fx0 + x0 / dfx0
    x_min = 0
    x_max = 0

    p = np.array([0, 0])

    if k_norm > 0:
        x_min = x0
        x_max = y2array[-1, 0]
        left = False

    elif k_norm < 0:
        x_min = y2array[0, 0]
        x_max = x0
        left = True

    else:
        print('zero')
        error = True

        # ДОБАВИТЬ ПОТОМ


    if left:
        sign_direction = -1
        y_direction = 1
    else:
        sign_direction = 0
        y_direction = -1

    direction = -1 if left else 0

    xrange = np.logical_and(x_min <= x, x_max >= x)
    y2_in_range = y2array[xrange]

    normalModel = Line(a=k_norm, c=z_norm)
    eq = normalModel.kanon_eq_vector(y2_in_range)
    eq[np.abs(eq) < eps] = 0.
    eq_sign = np.sign(eq)

    if np.all(eq_sign == eq_sign[0]):
        print('Пересечения нет')
        error = True
    else:
        eq_signchange = ((np.roll(eq_sign, y_direction) - eq_sign) != 0).astype(int)
        change_index = np.where(eq_signchange == 1)[0]

        if eq_sign[change_index[sign_direction]] == 0:
            p = y2_in_range[change_index[sign_direction]]
            print(p)
        else:
            a = y2_in_range[change_index[sign_direction] - y_direction]
            b = y2_in_range[change_index[sign_direction]]
            points = True

            k_segment = (b[1] - a[1]) / (b[0] - a[0])
            z_segment = (b[1] - a[1]) * (-a[0] / (b[0] - a[0]) + a[1] / (b[1] - a[1]))

            px = (z_segment - z_norm) / (k_norm - k_segment)
            py = (k_norm * z_segment - k_segment * z_norm) / (k_norm - k_segment)
            p = np.array([px, py])

            print(p)

if not error:
    fig, ax = plt.subplots(1)

    if inclination:
        y_min = np.amin(y2array[xrange, 1])
        y_max = np.amax(y2array[xrange, 1])
        normal = np.array([x[xrange], normalModel.y(x[xrange])]).T
        ax.plot(y2_in_range[:, 0], y2_in_range[:, 1])
        ax.plot(normal[:, 0], normal[:, 1])
        ax.scatter(y2array[:, 0], y2array[:, 1], color='black', s=4)
    else:
        ax.plot(y2array[:, 0], y2array[:, 1])
    #normalrange = np.logical_and(y_min - math.fabs(y_max - y_min) <= normal[:, 1], math.fabs(y_max - y_min) + y_max >= normal[:, 1])
    #normal = normal[normalrange]

    ax.xaxis.set_major_locator(ticker.MultipleLocator(0.5))
    ax.xaxis.set_minor_locator(ticker.MultipleLocator(0.1))
    ax.yaxis.set_major_locator(ticker.MultipleLocator(0.5))
    ax.yaxis.set_minor_locator(ticker.MultipleLocator(0.1))
    ax.grid(which='major')
    ax.grid(which='minor')

    ax.plot(x, y1)


    if points:
        ax.scatter(a[0], a[1])
        ax.scatter(b[0], b[1])

    ax.scatter(p[0], p[1])

    plt.show()
    # normal = np.array([x[xrange], normalModel.y(x[xrange])]).T
    # normalrange = np.logical_and(y_min - math.fabs(y_max - y_min) <= normal[:, 1], math.fabs(y_max - y_min) + y_max >= normal[:, 1])
    # normal = normal[normalrange]

    # if True in np.logical_and(y_min <= normal[:, 1], y_max >= normal[:, 1]):

    
        nbrs = NearestNeighbors(n_neighbors=1,
                                radius=0.1,
                                algorithm='auto',
                                metric='euclidean').fit(normal)

        distances, _ = nbrs.kneighbors(y2_in_range)
        # distances, indices = nbrs.kneighbors(y2_in_range)
        a_idx = np.argmin(distances, axis=0)
        a = y2_in_range[a_idx][0]
        res = normalModel.kanon_eq(a)
        

        if res == 0:
            p = a
        else:
            if res > 0:
                y2_half_plane = y2_in_range[normalModel.kanon_eq_vector(y2_in_range) < 0]

            else:
                y2_half_plane = y2_in_range[normalModel.kanon_eq_vector(y2_in_range) > 0]

            distances, _ = nbrs.kneighbors(y2_half_plane)
            b_idx = np.argmin(distances, axis=0)
            b = y2_half_plane[b_idx][0]

            k_segment = (b[1] - a[1]) / (b[0] - a[0])
            z_segment = (b[1] - a[1]) * (-a[0] / (b[0] - a[0]) + a[1] / (b[1] - a[1]))

            px = (z_segment - z_norm) / (k_norm - k_segment)
            py = (k_norm * z_segment - k_segment * z_norm) / (k_norm - k_segment)

            #px = (a[1] - a[0] * (b[1] - a[1]) / (b[0] - a[0]) + fx0 + x0 / dfx0) / \
            #     (-1./dfx0 - (b[1] - a[1]) / (b[0] - a[0]))

            #py = ((b[1] - a[1]) / (b[0] - a[0]) * ((a[0] + x0) / dfx0 + fx0) - a[1] / dfx0) / \
            #     (-1./dfx0 - (b[1] - a[1]) / (b[0] - a[0]))

            p = np.array([px, py])

            # if y2_half_plane.shape[0] == 0:
                # бряк

        ax.xaxis.set_major_locator(ticker.MultipleLocator(0.5))
        ax.xaxis.set_minor_locator(ticker.MultipleLocator(0.1))
        ax.yaxis.set_major_locator(ticker.MultipleLocator(0.5))
        ax.yaxis.set_minor_locator(ticker.MultipleLocator(0.1))
        ax.grid(which='major')
        ax.grid(which='minor')

        ax.plot(x, y1)
        ax.plot(y2_in_range[:, 0], y2_in_range[:, 1])
        ax.plot(normal[:, 0], normal[:, 1])

        ax.scatter(a[0], a[1])
        ax.scatter(b[0], b[1])
        ax.scatter(p[0], p[1])
        ax.scatter(y2array[:, 0], y2array[:, 1], color='black', s=4)

        plt.show()


# else:

norm_eq = Line(a=-1. / dfx0, c=fx0 + x0 / dfx0)
norm_vector = norm_eq.y(x)

kas = f_kas(fx0, dfx0, x0, x)
norm = f_norm(fx0, dfx0, x0, x)

print('kas-k = ', dfx0)
print('norm-k = ', -1. / dfx0)
print('kas-k - norm-k = ', dfx0 + 1. / dfx0)
print('kas-k * norm-k = ', dfx0 * (-1. / dfx0))
print('tg(gamma) = ', (dfx0 + 1. / dfx0) / (1 + dfx0 * (-1. / dfx0)))
print('gamma', math.atan((dfx0 + 1. / dfx0) / (1 + dfx0 * (-1. / dfx0))))
print('gamma in degrees = ', math.degrees(math.atan((dfx0 + 1. / dfx0) / (1 + dfx0 * (-1. / dfx0)))))

points_n = np.array([x, norm]).T

n_neighbors = 1  # сколько ближайших соседей хотим найти
nbrs = NearestNeighbors(n_neighbors=n_neighbors,
                        radius=0.1,
                        algorithm='auto',
                        metric='euclidean').fit(points_n)

points_y2 = np.array([x, y2]).T

distances, indices = nbrs.kneighbors(points_y2)
a_idx = np.argmin(distances, axis=0)
p = points_n[indices[a_idx][0][0]]

#
#for i, idx in enumerate(a_idx):
#    distance = distances[idx][i]
#    p1 = points_n[indices[idx][i]]
#    p2 = points_y2[idx]
#    print("Наименьшее расстояние между точками {} и {}, расстояние равно {}".format(p1, p2, distance))
#    ax.scatter(p1[0], p1[1])
#    ax.scatter(p2[0], p2[1])


ax.xaxis.set_major_locator(ticker.MultipleLocator(0.5))
ax.xaxis.set_minor_locator(ticker.MultipleLocator(0.1))
ax.yaxis.set_major_locator(ticker.MultipleLocator(0.5))
ax.yaxis.set_minor_locator(ticker.MultipleLocator(0.1))
ax.grid()

ax.plot(x, y1)
ax.plot(x, y2)
ax.plot(x, kas)
ax.plot(x, norm)
ax.scatter(p[0], p[1])
ax.scatter(x[ind], y1[ind])

print('distance = ', math.sqrt((p[0] - x[ind]) ** 2 + (p[1] - y1[ind]) ** 2))

'''
unit = 0.5
x_tick = np.arange(-0.1, 0.1+unit, unit)
x_label = [r"$-\frac{\pi}{2}$", r"$-\frac{\pi}{4}$", r"$0$", r"$+\frac{\pi}{4}$",   r"$+\frac{\pi}{2}$"]
ax.set_xticks(x_tick*np.pi)
ax.set_xticklabels(x_label, fontsize=12)
'''

plt.show()

"""
