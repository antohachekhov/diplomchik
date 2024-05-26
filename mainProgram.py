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

    def __init__(self, signalForImage, signalForCurrentStep):
        self.image = None
        self.mutex = threading.Lock()
        self.nameSharedMemory = 'curImage'
        self.sharedMemory = None
        self.estimation = None
        self.threadProg = None
        self.threadShow = None
        self.signalForImage = signalForImage
        self.signalForCurrentStep = signalForCurrentStep

    def inputImage(self, path: str):
        try:
            InputImage = PILImage.open(path).convert('L')
            self.image = np.asarray(InputImage.crop((0, 0, InputImage.size[0], InputImage.size[1] - 103)))
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
            print('Поток 1: Начинаю')
            result, _ = self.estimation.estimateTransitionLayer(self.mutex, self.nameSharedMemory,
                                                                self.signalForCurrentStep)
            print('Поток 1: Получены результаты')
            print(result.shape)
            self.estimation.clear()
            print(result)
            print(np.median(list(itertools.chain(*result))))
            print('Поток 1: Заканчиваю')
            return np.median(list(itertools.chain(*result)))
        except Exception as e:
            print(f'Поток 1: жёсткая ошибка <<{e}>>')
            print('Поток 1: завершил работу')

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
                # cv.imshow('data', datacopy)
                # cv.waitKey(3000)
                # cv.destroyAllWindows()
                h, w = datacopy.shape[: 2]
                qImg = QImage(datacopy, w, h, 3 * w, QImage.Format_RGB888)
                self.signalForImage.emit(qImg)
                self.mutex.release()
                time.sleep(1)
                print(f'{threading.current_thread().name} освободил ресурс')

        # self.sharedMemory.close()
        # self.sharedMemory.unlink()
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
# fieldPath = r"C:\Users\bortn\Desktop\diplomchik\analysis\new dataset\5\1-11\field_filt_prism_w30x1y1.csv"
fieldPath = r"C:\Users\bortn\Desktop\diplomchik\analysis\new dataset\20\1-04\field_field_prism_w30x1y1.csv"

# threadProg = None
# threadShow = None

# if __name__ == "__main__":
#     img = inputImageWithCrop(imageName)
#     field = inputField(fieldPath)
#     field = None
#
#     mutex = threading.Lock()
#     nameSharedMem = 'curImage'
#     sizeOfMemory = cv.cvtColor(img, cv.COLOR_GRAY2RGB).nbytes
#     sharedMemory = SharedMemory(name=nameSharedMem, size=sizeOfMemory, create=True)
#
#     threadProg = threading.Thread(target=main, args=(img, field, mutex, nameSharedMem))
#     threadShow = threading.Thread(target=showImage,
#                                   args=(cv.cvtColor(img, cv.COLOR_GRAY2RGB).shape, mutex, nameSharedMem))
#     threadProg.start()
#     threadShow.start()
#     threadProg.join()
#     sharedMemory.close()
#     sharedMemory.unlink()
