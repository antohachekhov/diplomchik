from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtGui import QPixmap
from PyQt5.QtWidgets import QMainWindow, QApplication, QLabel, QPushButton, QTextEdit
from mainProgram import InterfaceProgram

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
    gotoSignal = QtCore.pyqtSignal(str)

    def goto(self, name):
        self.gotoSignal.emit(name)


class MainWindow(PageWindow):
    signalToChangeImage = QtCore.pyqtSignal(QtGui.QImage)
    signalStepOfAnalysis = QtCore.pyqtSignal(str)

    def __init__(self):
        super().__init__()
        self.initUI()
        self.setWindowTitle("Измерение переходного слоя")
        self.signalToChangeImage.connect(self.changeImage)
        self.signalStepOfAnalysis.connect(self.changeTextField)
        self.program = InterfaceProgram(self.signalToChangeImage, self.signalStepOfAnalysis)

    def changeImage(self, image):
        qPixmap = QPixmap.fromImage(image).scaledToHeight(self.image.height())
        self.image.setPixmap(qPixmap)

    def changeTextField(self, message):
        self.textField.setText(message)

    def initUI(self):
        self.UiComponents()

    def UiComponents(self):
        self.settingsButton = QPushButton("Настройки", self)
        self.settingsButton.setGeometry(QtCore.QRect(5, 5, 150, 30))
        self.settingsButton.setObjectName('changePage')
        self.settingsButton.clicked.connect(
            self.make_handleButton("settingsButton")
        )

        self.image = QLabel(self)
        self.image.setObjectName('image')
        self.image.setGeometry(QtCore.QRect(89, 50, 768, 500))

        self.buttonInput = QPushButton("Загрузить изображение", self.image)
        self.buttonInput.setObjectName('button')
        self.buttonInput.setGeometry(301, 230, 167, 39)
        self.buttonInput.clicked.connect(self.inputImage)

        self.textField = QTextEdit(self)
        self.textField.setGeometry(89, 561, 768, 60)
        self.textField.setObjectName('textField')
        self.textField.setReadOnly(True)
        self.textField.setText(
            'Привет, мир!\nПривет, Никита!\nПривет, мир!\nПривет, Никита!\nПривет, мир!\nПривет, Никита!')

        self.startButton = QPushButton('Начать анализ', self)
        self.startButton.setGeometry(296, 640, 167, 35)
        self.startButton.setObjectName('button')
        self.startButton.setEnabled(False)
        self.startButton.clicked.connect(self.startAnalysis)

        self.clear = QPushButton('Очистить', self)
        self.clear.setGeometry(483, 640, 167, 35)
        self.clear.setObjectName('button')
        self.clear.setEnabled(False)
        self.clear.clicked.connect(self.clearClick)
        #pixmap = QPixmap('test.jpg').scaledToHeight(500)
        #self.image.setPixmap(pixmap)
        # self.setCentralWidget(self.image)
        # self.image.move(70, 84)

        # container = QtWidgets.QVBoxLayout()
        # container.addWidget(self.settingsButton)
        # container.addWidget(self.image)
        # self.setLayout(container)

    def inputImage(self):
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

    def startAnalysis(self):
        self.program.estimate()

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

        self.setFixedSize(945, 715)

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
    import sys

    app = QtWidgets.QApplication(sys.argv)
    app.setStyleSheet(styleSheets)
    w = Window()
    w.show()
    sys.exit(app.exec_())
