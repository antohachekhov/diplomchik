import threading
from testNewProgram import main
from multiprocessing.shared_memory import SharedMemory
import cv2 as cv
from PIL import Image as PILImage
import numpy as np

def showImage(shape, mutex, nameSharedMemory):
    global threadProg
    sharedMemory = SharedMemory(name=nameSharedMemory, create=False)
    data = np.ndarray(shape, dtype=np.uint8, buffer=sharedMemory.buf)
    while threadProg.is_alive():
        print(f'{threading.current_thread().name} ожидает доступ к ресурсу')
        mutex.acquire()
        print(f'{threading.current_thread().name} получил доступ к ресурсу')
        if data is None:
            mutex.release()
            print(f'{threading.current_thread().name} освободил ресурс. Он был пуст')
            break
        else:
            cv.imshow('data', data)
            cv.waitKey(3000)
            cv.destroyAllWindows()
            mutex.release()
        print(f'{threading.current_thread().name} освободил ресурс')

def inputImageWithCrop(path:str):
    InputImage = PILImage.open(path).convert('L')
    return np.asarray(InputImage.crop((0, 0, InputImage.size[0], InputImage.size[1] - 103)))

def inputField(path:str):
    return np.loadtxt(path, delimiter=";")

imageName = r"C:\Users\bortn\Desktop\diplomchik\analysis\new dataset\5\1-11\1-11.tif"
fieldPath = r"C:\Users\bortn\Desktop\diplomchik\analysis\new dataset\5\1-11\field_filt_prism_w30x1y1.csv"

threadProg = None
threadShow = None

if __name__ == "__main__":
    img = inputImageWithCrop(imageName)
    field = inputField(fieldPath)

    mutex = threading.Lock()
    nameSharedMem = 'curImage'
    sizeOfMemory = cv.cvtColor(img, cv.COLOR_GRAY2RGB).nbytes
    sharedMemory = SharedMemory(name=nameSharedMem, size=sizeOfMemory, create=True)

    threadProg = threading.Thread(target=main, args=(img, field, mutex, nameSharedMem))
    threadShow = threading.Thread(target=showImage, args=(cv.cvtColor(img, cv.COLOR_GRAY2RGB).shape, mutex, nameSharedMem))
    threadProg.start()
    threadShow.start()
    threadProg.join()
    sharedMemory.close()
    sharedMemory.unlink()