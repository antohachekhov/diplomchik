from measureObjectOfBinaryImage import *

hsv_min = 254
hsv_max = 255

if __name__ == '__main__':
    # круг - 600
    image = cv.imread(r"C:\Users\bortn\Desktop\diplomchik\testMeasure2\circle600.jpg", cv.IMREAD_GRAYSCALE)

    thresh = cv.inRange(image, hsv_min, hsv_max)
    contours0, hierarchy = cv.findContours(thresh, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_NONE)
    win_size = 0

    measurer = Measure()
    measurer(image, thresh, win_size)