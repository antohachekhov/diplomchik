import sys
import numpy as np
import multiprocessing

from PyQt5.QtGui import QPixmap
from mainProgram import InterfaceProgram
from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtWidgets import QLabel, QPushButton, QTextEdit

styleSheets = """
    QPushButton#changePage {
        background-color: white;
        border-radius: 10px;
        border: 1px solid #8ED6FF;
    }

    QPushButton#changePage:hover {
        border: 2px solid #8ED6FF;
        background-color: #E6F6FF;
    }

    QPushButton#changePage:pressed {
        background-color: #8ED6FF;
    }

    QLabel#image { 
        background-color: #E6F6FF;
    }

    QMainWindow {
        background-color: white;
    }

    QPushButton#button {
        background-color: #8ED6FF;
        color: black;
        border-radius: 12px;
    }

    QPushButton#button:hover {
        background: qlineargradient(x1:0, y1:0, x2:0, y2:1, stop:0 #8ED6FF,stop:1 #1AACFF);
    }

    QPushButton#button:pressed {
        background-color: #1AACFF;
    }

    QPushButton#button:disabled {
        background-color: #C7C7C7;
    }

    QTextEdit#textField {
        border: 1px solid #8ED6FF;
    }
"""


class PageWindow(QtWidgets.QMainWindow):
    """
    Класс страницы окна
    """
    gotoSignal = QtCore.pyqtSignal(str)

    def goto(self, name):
        self.gotoSignal.emit(name)


class MainWindow(PageWindow):
    """
    Класс главного окна
    """

    class TableOfResults(QtWidgets.QTableWidget):
        """
        Класс таблицы результатов
        """

        def __init__(self, parent=None, slotForChange=None):
            super().__init__(parent)
            self.slotForChange = slotForChange
            self.setColumnCount(2)
            self.setRowCount(3)

        def showResult(self, results):
            """
            Отображение результатов
            """
            self.setRowCount(len(results.result))
            for row in range(self.rowCount()):
                checkBox = QtWidgets.QCheckBox(self)
                checkBox.setChecked(True)
                if self.slotForChange is not None:
                    checkBox.clicked.connect(self.slotForChange)
                self.setCellWidget(row, 0, checkBox)
                valueCell = f"{np.median(results.result[row]) * results.scaleCoef:.4g} {results.unit if results.unit is not None else 'pixels'}"
                cell = QtWidgets.QTableWidgetItem(valueCell)
                self.setItem(row, 1, cell)
            self.resizeColumnsToContents()

    signalToChangeImage = QtCore.pyqtSignal(QtGui.QImage)
    signalStepOfAnalysis = QtCore.pyqtSignal(str)
    signalResult = QtCore.pyqtSignal(bool)

    def __init__(self):
        """
        Конструктор класса
        """
        super().__init__()
        self.initUI()
        self.initLogic()
        self.setWindowTitle("Измерение переходного слоя")

    def initUI(self):
        """
        Инициализация графических элементов окна
        """
        # self.settingsButton = QPushButton("Настройки", self)
        # self.settingsButton.setGeometry(QtCore.QRect(5, 5, 150, 30))
        # self.settingsButton.setObjectName('changePage')
        # self.settingsButton.clicked.connect(self.make_handleButton("settingsButton"))

        self.image = QLabel(self)
        self.image.setObjectName('image')
        self.image.setGeometry(QtCore.QRect(24, 50, 768, 500))

        self.buttonInput = QPushButton("Загрузить изображение", self.image)
        self.buttonInput.setObjectName('button')
        self.buttonInput.setGeometry(300, 230, 167, 39)
        self.buttonInput.clicked.connect(self.inputImage)

        self.textFieldLabel = QLabel("Статус анализа", self)
        self.textFieldLabel.setObjectName('label')
        self.textFieldLabel.setGeometry(24, 552, 768, 18)

        self.fieldCurrentStatus = QTextEdit(self)
        self.fieldCurrentStatus.setGeometry(24, 570, 768, 60)
        self.fieldCurrentStatus.setObjectName('textField')
        self.fieldCurrentStatus.setReadOnly(True)
        self.fieldCurrentStatus.setText(
            'Выберите изображение для анализа. Для этого нажмите кнопку "Загрузить изображение". После выбора снимка сплава нажмите на кнопку "Начать анализ".')

        self.startButton = QPushButton('Начать анализ', self)
        self.startButton.setGeometry(231, 640, 167, 35)
        self.startButton.setObjectName('button')
        self.startButton.setEnabled(False)
        self.startButton.clicked.connect(self.startAnalysis)

        self.clear = QPushButton('Очистить', self)
        self.clear.setGeometry(418, 640, 167, 35)
        self.clear.setObjectName('button')
        self.clear.setEnabled(False)
        self.clear.clicked.connect(self.clearClick)

        # self.table = QtWidgets.QTableWidget(self)
        # self.table.setColumnCount(2)  # Set three columns
        # self.table.setRowCount(3)
        # self.table.setGeometry(QtCore.QRect(813, 50, 203, 500))
        self.table = self.TableOfResults(self, self.checkBoxChanged)
        self.table.setGeometry(QtCore.QRect(813, 50, 203, 500))

        self.resultLabel = QLabel("Общее значение", self)
        self.resultLabel.setObjectName('label')
        self.resultLabel.setGeometry(813, 552, 203, 18)

        self.resultField = QTextEdit(self)
        self.resultField.setGeometry(813, 570, 203, 60)
        self.resultField.setObjectName('textField')
        self.resultField.setReadOnly(True)
        self.resultField.setText('')

    def initLogic(self):
        """
        Инициализация программы оценки ширины переходного слоя
        :return:
        """
        self.signalToChangeImage.connect(self.changeImage)
        self.signalStepOfAnalysis.connect(self.putResultsOnTextField)
        self.signalResult.connect(self.processingResult)

        self.program = InterfaceProgram()
        self.program.setSignals(self.signalToChangeImage, self.signalStepOfAnalysis, self.signalResult)

    def changeImage(self, image):
        """
        Изменение изображения в окне
        :param image:
        :return:
        """
        qPixmap = QPixmap.fromImage(image).scaledToHeight(self.image.height())
        self.image.setPixmap(qPixmap)

    def putResultsOnTextField(self, message):
        """
        Вывод сообщения в текстовое поле
        """
        self.fieldCurrentStatus.setText(message)

    def processingResult(self, flagOfEnd):
        """
        Обработка результатов анализа
        """
        if flagOfEnd:
            self.table.showResult(self.program.currentResult)
            self.resultField.setText(self.program.currentResult.getTotalWidth([True] * len(self.program.currentResult.result)))
            self.fieldCurrentStatus.setText('Анализ завершён')
            self.clear.setEnabled(True)
        else:
            raise self.fieldCurrentStatus.setText('Произошла неизвестная ошибка')

    def checkBoxChanged(self):
        """
        Функция, выполняющаяся при изменении check-box в таблице
        :return:
        """
        flags = [True] * len(self.program.currentResult.result)
        for i in range(self.table.rowCount()):
            flags[i] = self.table.cellWidget(i, 0).isChecked()
        self.resultField.setText(self.program.currentResult.getTotalWidth(flags))

    def inputImage(self):
        """
        Ввод изображения
        """
        fileName = QtWidgets.QFileDialog.getOpenFileNames(self, "Open Image", "./", "*.tif")[0]
        if len(fileName):
            if self.program.inputImage(fileName[0]):
                self.startButton.setEnabled(True)
                self.clear.setEnabled(True)
                self.buttonInput.hide()

    def clearClick(self):
        self.program.clear()
        self.image.clear()
        self.buttonInput.show()
        self.startButton.setEnabled(False)
        self.clear.setEnabled(False)
        self.fieldCurrentStatus.clear()
        self.table.clear()
        self.resultField.clear()

    def startAnalysis(self):
        """
        Функция, выполняющая при нажатии кнопки "Начать анализ"
        :return:
        """
        self.program.estimate()
        self.startButton.setEnabled(False)
        self.clear.setEnabled(False)

    def make_handleButton(self, button):
        def handleButton():
            if button == "settingsButton":
                self.goto("settings")

        return handleButton


class SettingsWindow(PageWindow):
    def __init__(self):
        super().__init__()
        self.initUI()

    def initUI(self):
        self.setWindowTitle("Измерение переходного слоя - Настройки")
        self.UiComponents()

    def goToMain(self):
        self.goto("main")

    def UiComponents(self):
        self.backButton = QPushButton("Вернуться на главную", self)
        self.backButton.setGeometry(QtCore.QRect(5, 5, 150, 30))
        self.backButton.setObjectName('changePage')
        self.backButton.clicked.connect(self.goToMain)


class Window(QtWidgets.QMainWindow):
    def __init__(self, parent=None):
        super().__init__(parent)

        self.setFixedSize(1040, 715)

        self.stacked_widget = QtWidgets.QStackedWidget()
        self.setCentralWidget(self.stacked_widget)

        self.m_pages = {}

        self.register(MainWindow(), "main")
        self.register(SettingsWindow(), "settings")

        self.goto("main")

    def register(self, widget, name):
        self.m_pages[name] = widget
        self.stacked_widget.addWidget(widget)
        if isinstance(widget, PageWindow):
            widget.gotoSignal.connect(self.goto)

    @QtCore.pyqtSlot(str)
    def goto(self, name):
        if name in self.m_pages:
            widget = self.m_pages[name]
            self.stacked_widget.setCurrentWidget(widget)
            self.setWindowTitle(widget.windowTitle())


if __name__ == "__main__":
    multiprocessing.freeze_support()
    app = QtWidgets.QApplication(sys.argv)
    app.setStyleSheet(styleSheets)
    w = Window()
    w.show()
    sys.exit(app.exec_())
