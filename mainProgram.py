import threading
import time

from testNewProgram import main
from multiprocessing.shared_memory import SharedMemory
import cv2 as cv
from PIL import Image as PILImage
import numpy as np
from copy import deepcopy
import itertools
from ProgramEstimationTransitionLayer import EstimationTransitionLayer
from PyQt5.QtWidgets import QLabel
from PyQt5.QtGui import QImage, QPixmap


class InterfaceProgram:

    def __init__(self, signalForImage, signalForCurrentStep, signalForResult):
        self.image = None
        self.mutex = threading.Lock()
        self.nameSharedMemory = 'curImage'
        self.sharedMemory = None
        self.estimation = None
        self.threadProg = None
        self.threadShow = None
        self.signalForImage = signalForImage
        self.signalForCurrentStep = signalForCurrentStep
        self.signalForResult = signalForResult
        self.scaleCoef = None
        self.unit = 'pixel'

    def inputImage(self, path: str):
        try:
            InputImage = PILImage.open(path).convert('L')
            self.image = np.asarray(InputImage.crop((0, 0, InputImage.size[0], InputImage.size[1] - 103)))

            from ImagesMetadata import getScale
            try:
                self.scaleCoef, self.unit = getScale(path)
                self.unit += 'm'
            except:
                self.scaleCoef = None
        except:
            return False
        else:
            self.estimation = EstimationTransitionLayer(False)
            self.estimation.setSettings(WindowProcessing={'parallelComputing': True, 'windowSize': 30})
            self.estimation.setImage(self.image, inputField(fieldPath))

            h, w = self.image.shape[: 2]
            qImg = QImage(self.image, w, h, w, QImage.Format_Grayscale8)
            self.signalForImage.emit(qImg)
            return True

    def estimate(self):
        sizeOfMemory = cv.cvtColor(self.image, cv.COLOR_GRAY2RGB).nbytes
        self.sharedMemory = SharedMemory(name=self.nameSharedMemory, size=sizeOfMemory, create=True)
        print('Создана общая память')
        self.threadProg = threading.Thread(target=self._programStart)
        print('1 поток инициализирован')
        self.threadShow = threading.Thread(target=self._showImage)
        print('2 поток инициализирован')
        self.threadProg.start()
        self.threadShow.start()
        print('Потоки запущены')
        # self.threadProg.join()
        # self.sharedMemory.close()
        # self.sharedMemory.unlink()
        # self.sharedMemory = None
        # self.threadProg = None
        # self.threadShow = None

    def clear(self):
        self.estimation.clear()
        self.image = None
        if self.sharedMemory is not None:
            self.sharedMemory.close()
            self.sharedMemory.unlink()
            self.sharedMemory = None

    def _programStart(self):
        try:
            print(f'{threading.current_thread().name} Начинаю')
            result = self.estimation.estimateTransitionLayer(self.mutex, self.nameSharedMemory,
                                                                self.signalForCurrentStep)
            print(f'{threading.current_thread().name} Получены результаты')
            self.estimation.clear()
            # results = [np.median(obj) for obj in result]
            # results.append(np.median(list(itertools.chain(*result))))
            # if self.scaleCoef is not None:
            #     resultsStr = [str(val * self.scaleCoef) + self.unit for val in results]
            # for i, val in enumerate(resultsStr[:-1]):
            #     resultsStr[i] = f'Объект {i + 1}: ' + val
            # resultsStr[-1] = 'Общее значение: ' + resultsStr[-1]
            # print(np.median(list(itertools.chain(*result))))
            print(f'{threading.current_thread().name} Отправляю результаты')
            self.signalForResult.emit(result)
            # self.signalForCurrentStep.emit('\n'.join(resultsStr))
            print(f'{threading.current_thread().name} Заканчиваю')
            return True
        except Exception as e:
            print(f'{threading.current_thread().name} ОШИБКА "{e}"')
            print(f'{threading.current_thread().name} завершил работу')

    def _showImage(self):
        shape = cv.cvtColor(self.image, cv.COLOR_GRAY2RGB).shape
        sharedMemory = SharedMemory(name=self.nameSharedMemory, create=False)
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
                if self.signalForImage is not None:
                    h, w = datacopy.shape[: 2]
                    qImg = QImage(datacopy, w, h, 3 * w, QImage.Format_RGB888)
                    self.signalForImage.emit(qImg)
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
        self.sharedMemory = None
        print('Поток 2: завершил работу')


def inputImageWithCrop(path: str):
    InputImage = PILImage.open(path).convert('L')
    return np.asarray(InputImage.crop((0, 0, InputImage.size[0], InputImage.size[1] - 103)))


def inputField(path: str):
    return np.loadtxt(path, delimiter=";")


imageName = None
fieldPath = None
# imageName = r"C:\Users\bortn\Desktop\diplomchik\analysis\new dataset\5\1-11\1-11.tif"
fieldPath = r"C:\Users\bortn\Desktop\diplomchik\analysis\new dataset\5\1-11\field_filt_prism_w30x1y1.csv"
# fieldPath = r"C:\Users\bortn\Desktop\diplomchik\analysis\new dataset\20\1-04\field_field_prism_w30x1y1.csv"

