import time
import cv2 as cv
import threading
import itertools
import statistics
import numpy as np

from copy import deepcopy
from PyQt5.QtGui import QImage
from PIL import Image as PILImage
from multiprocessing.shared_memory import SharedMemory
from ProgramEstimationTransitionLayer import EstimationTransitionLayer


class InterfaceProgram:
    """
    Класс основной программы

    Атрибуты
    ----------
    currentImage : ndarray, по умолчанию = None
        Текущее введённое изображение

    currentResult : ResultsOfAnalysis, по умолчанию = None
        Текущие результаты анализа

    mutex : threading.Lock, по умолчанию = threading.Lock()
        Мутекс для синхронизации потоков

    sharedMemory : SharedMemory, по умолчанию = None
        Общая память

    nameSharedMemory : str, по умолчанию = 'curImage'
        Название общей памяти

    threadProg : threading.Thread, по умолчанию = None
        Поток оценки ширины переходного слоя

    threadImage : threading.Thread, по умолчанию = None
        Поток обработки промежуточных результатов

    GUIconnect : bool, по умолчанию = False
        Флаг, сообщающий о соединении с графическим интерфейсом

    estimator : EstimationTransitionLayer, по умолчанию = EstimationTransitionLayer()
        Модуль оценки ширины переходного слоя
    """
    class ResultsOfAnalysis:
        """
        Класс результатов анализа изображения

        Атрибуты
        ----------
        result : list, по умолчанию = None
            Результаты измерения ширины каждой сегментированной области в каждой точке её контура

        scaleCoef : float
            Коэффициент масштабирования

        unit : str, по умолчанию = 'm'
            Единица измерения результатов
        """
        measurementDict = {
            "m": 1E-3,
            "µ": 1E-6,
            "n": 1E-9,
        }

        def __init__(self, scaleCoef, scaleCoefFromUnit='', unit='m'):
            """
            Конструктор класса
            :param scaleCoef: float
                Коэффициент масштабирования
            :param scaleCoefFromUnit: str
                Единица измерения до изменения масштаба
            :param unit: str
                Единица измерения после изменения масштаба
            """
            self.result = None
            self.scaleCoef = scaleCoef
            if unit == 'm':
                self.scaleCoef *= self.measurementDict[scaleCoefFromUnit]
            elif unit != 'pixels':
                raise Exception(f'Такая единица измерения пока не поддерживается ({unit})')
            self.unit = unit

        def setResult(self, results):
            """
            Ввод результатов анализа изображения
            :param results: list
                Результаты анализа
            :return: None
            """
            self.result = results

        def getTotalWidth(self, flags):
            """
            Оценка общей ширины переходного слоя по заданным объектам
            :param flags: list
                Список, где элемент указывает на использование
                соответствующего объекта при общей оценке переходного слоя (True/False)
            :return: str
                Строковое представление результата общей оценки ширины
                переходного слоя
            """
            if len(flags) != len(self.result):
                raise Exception('Количество меток не равно количеству объектов')
            if any(flags):
                valuesForTotalValue = []
                for i, values in enumerate(self.result):
                    if flags[i]:
                        valuesForTotalValue.append(values)
                union = np.array(list(itertools.chain(*valuesForTotalValue)))
                medianFull = np.median(union)
                quartiles = statistics.quantiles(union)
                coef = 1.5
                maxLimin = quartiles[2] + coef * (quartiles[2] - quartiles[0])
                minLimin = quartiles[0] - coef * (quartiles[2] - quartiles[0])
                medianWithoutOutlier = np.median(union[np.logical_and(minLimin <= union, union <= maxLimin)])
                if self.scaleCoef:
                    medianFull *= self.scaleCoef
                    medianWithoutOutlier *= self.scaleCoef
                # Без выбросов: {medianWithoutOutlier:.4g} {self.program.unit}"
                return f"{medianWithoutOutlier:.4g} {self.unit}"
            else:
                return 'Не выбрано ни одного объекта'

        def clear(self):
            """
            Очистка полей объекта класса
            :return: None
            """
            self.result = None
            self.scaleCoef = None
            self.unit = None

    # Сигналы для графического интерфейса
    signalsForGUI = {
        'signalForImage': None,
        'signalForCurrentStep': None,
        'signalForResult': None
    }

    def __init__(self):
        """
        Конструктор класса
        """
        self.currentImage = None
        self.currentResult = None

        self.mutex = threading.Lock()
        self.nameSharedMemory = 'curImage'
        self.threadProg = None
        self.threadImage = None
        self.sharedMemory = None

        self.GUIconnect = False

        self.estimator = EstimationTransitionLayer(False)
        self.estimator.setSettings(WindowProcessing={'parallelComputing': True},
                                   Multithreading={'SignalForLoggingOfCurrentStep': self.signalsForGUI['signalForCurrentStep'],
                                                   'NameOfSharedMemory': self.nameSharedMemory,
                                                   'Mutex': self.mutex})

    def setSignals(self, signalForImage, signalForCurrentStep, signalForResult):
        """
        Установка сигналов для графического интерфейса
        :param signalForImage: QSignal
            Сигнал о смене изображения
        :param signalForCurrentStep: QSignal
            Сигнал о текущем шага выполнения алгоритма
        :param signalForResult: QSignal
            Сигнал о готовности результатов
        :return:
        """
        self.signalsForGUI['signalForImage'] = signalForImage
        self.signalsForGUI['signalForCurrentStep'] = signalForCurrentStep
        self.signalsForGUI['signalForResult'] = signalForResult
        self.GUIconnect = True

    def sendImage(self, image, format='gray'):
        """
        Отправка изображения в графический интерфейс
        :param image: ndarray
            Изображение
        :param format: str
            Указывает на цветовое пространство изображения
        :return:
        """
        h, w = image.shape[: 2]
        if format == 'gray':
            qImg = QImage(image, w, h, w, QImage.Format_Grayscale8)
        elif format == 'rgb':
            qImg = QImage(image, w, h, w * 3, QImage.Format_RGB888)
        else:
            raise Exception('Неизвестный формат изображения')
        self.signalsForGUI['signalForImage'].emit(qImg)

    def inputImage(self, path: str):
        """
        Ввод изображения
        :param path: str
            Путь к изображению
        :return: bool
            True - если ввод произошёл успешно, иначе - False
        """
        try:
            InputImage = PILImage.open(path).convert('L')
            self.currentImage = np.asarray(InputImage.crop((0, 0, InputImage.size[0], InputImage.size[1] - 103)))
            try:
                from ImagesMetadata import getScale
                scaleCoef, scaleCoefToUnit = getScale(path)
                self.currentResult = self.ResultsOfAnalysis(scaleCoef, scaleCoefToUnit, 'm')
            except Exception:
                self.currentResult = self.ResultsOfAnalysis(1.0, unit='pixels')
        except Exception:
            return False
        else:
            self.estimator.setImage(self.currentImage, inputField(fieldPath) if fieldPath is not None else None)
            self.sendImage(self.currentImage)
            return True

    def estimate(self):
        """
        Начать анализ изображения
        :return: None
        """
        # Определение размера общей памяти
        sizeOfMemory = cv.cvtColor(self.currentImage, cv.COLOR_GRAY2RGB).nbytes
        # Инициализация общей памяти
        self.sharedMemory = SharedMemory(name=self.nameSharedMemory, size=sizeOfMemory, create=True)
        # Инициализация потоков
        print('Создана общая память')
        self.threadProg = threading.Thread(target=self._programStart)
        print('1 поток инициализирован')
        self.threadImage = threading.Thread(target=self._showImage)
        print('2 поток инициализирован')
        # Запуск потоков
        self.threadProg.start()
        self.threadImage.start()
        print('Потоки запущены')
        # self.threadProg.join()
        # self.sharedMemory.close()
        # self.sharedMemory.unlink()

    def clear(self):
        """
        Очистка полей объекта класса
        :return: None
        """
        self.estimator.clear(False)
        self.currentResult.clear()
        self.currentImage = None
        if self.sharedMemory is not None:
            self.sharedMemory.close()
            self.sharedMemory.unlink()
            self.sharedMemory = None

    def _programStart(self):
        """
        Функция потока, анализирующего изображение
        :return: bool
            True - если всё прошло успешно, иначе - False
        """
        try:
            print(f'{threading.current_thread().name} Начинаю')
            result = self.estimator.estimateTransitionLayer()
            print(f'{threading.current_thread().name} Отправляю результаты')
            self.currentResult.setResult(result)
            if self.GUIconnect:
                self.signalsForGUI['signalForResult'].emit(True)
            self.estimator.clear()
            print(f'{threading.current_thread().name} Заканчиваю')
            return True
        except Exception as e:
            if self.GUIconnect:
                self.signalsForGUI['signalForResult'].emit(False)
            self.estimator.clear()
            print(f'{threading.current_thread().name} ОШИБКА "{e}"')
            print(f'{threading.current_thread().name} завершил работу')

    def _showImage(self):
        """
        Функция потока, обрабатывающего промежуточные результаты
        :return: None
        """
        shape = cv.cvtColor(self.currentImage, cv.COLOR_GRAY2RGB).shape
        sharedMemory = self.sharedMemory # SharedMemory(name=self.nameSharedMemory, create=False)
        data = np.ndarray(shape, dtype=np.uint8, buffer=sharedMemory.buf)
        while self.threadProg.is_alive():
            print(f'{threading.current_thread().name} ожидает доступ к ресурсу')
            self.mutex.acquire()
            print(f'{threading.current_thread().name} получил доступ к ресурсу')
            if sharedMemory.buf is None:
                self.mutex.release()
                print(f'{threading.current_thread().name} освободил ресурс. Он был пуст')
                break
            else:
                datacopy = deepcopy(data)
                if self.signalsForGUI['signalForImage'] is not None:
                    self.sendImage(datacopy, 'rgb')
                else:
                    cv.imshow('data', datacopy)
                    cv.waitKey(3000)
                    cv.destroyAllWindows()
                self.mutex.release()
                time.sleep(1)
                print(f'{threading.current_thread().name} освободил ресурс')

        if sharedMemory.buf is not None:
            self.sharedMemory.close()
            self.sharedMemory.unlink()
            print(f'{threading.current_thread().name} закрыл общую память')
        self.sharedMemory = None
        print(f'{threading.current_thread().name} завершил работу')


def inputImageWithCrop(path: str):
    InputImage = PILImage.open(path).convert('L')
    return np.asarray(InputImage.crop((0, 0, InputImage.size[0], InputImage.size[1] - 103)))


def inputField(path: str):
    return np.loadtxt(path, delimiter=";")

nComp = None
imageName = None
fieldPath = None
# imageName = r"C:\Users\bortn\Desktop\diplomchik\analysis\new dataset\5\1-11\1-11.tif"
# fieldPath = r"C:\Users\bortn\Desktop\diplomchik\analysis\new dataset\25\1-40\field_filt25_180_prism_w30x1y1.csv"
# fieldPath = r"C:\Users\bortn\Desktop\diplomchik\analysis\new dataset\5\1-11\field_filt_prism_w30x1y1.csv"
fieldPath = r"C:\Users\bortn\Desktop\diplomchik\analysis\new dataset\20\1-04\field_prism_w30x1y1.csv"
# fieldPath = r"C:\Users\bortn\Desktop\diplomchik\analysis\new dataset\20\1-34\field_filt_w30.csv"

