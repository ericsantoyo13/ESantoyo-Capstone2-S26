import sys
from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QApplication, QWidget, QVBoxLayout, QHBoxLayout,
    QPushButton, QLabel, QStackedWidget
)


class MenuScreen(QWidget):
    def __init__(self, go_screen1, go_screen2, go_screen3, go_screen4):
        super().__init__()
        layout = QVBoxLayout(self)
        layout.setSpacing(12)

        title = QLabel("Main Menu")
        title.setAlignment(Qt.AlignCenter)
        layout.addWidget(title)

        btn1 = QPushButton("Go to Keyring Sizing")
        btn1.clicked.connect(go_screen1)
        layout.addWidget(btn1)

        btn2 = QPushButton("Go to U.S. Coin Selection")
        btn2.clicked.connect(go_screen2)
        layout.addWidget(btn2)

        btn3 = QPushButton("Go to some other screen that will be unused")
        btn3.clicked.connect(go_screen3)
        layout.addWidget(btn3)

        btn4 = QPushButton("Go to another unused screen")
        btn4.clicked.connect(go_screen4)
        layout.addWidget(btn4)

        layout.addStretch(1)


class SimpleScreen(QWidget):
    def __init__(self, name: str, go_menu):
        super().__init__()
        layout = QVBoxLayout(self)
        layout.setSpacing(12)

        title = QLabel(name)
        title.setAlignment(Qt.AlignCenter)
        layout.addWidget(title)

        layout.addWidget(QLabel(f"This is the page dedicated to {name}."))
        layout.addStretch(1)

        back_btn = QPushButton("Return to Menu")
        back_btn.clicked.connect(go_menu)
        layout.addWidget(back_btn)


class MainWindow(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Welcome to the main menu!")
        self.resize(600, 400)

        self.stack = QStackedWidget()

        root = QVBoxLayout(self)
        root.addWidget(self.stack)

        self.screen1 = SimpleScreen("Keyring Sizing", self.show_menu)
        self.screen2 = SimpleScreen("Selecting U.S. Coin to use as reference", self.show_menu)
        self.screen3 = SimpleScreen("Arbitrary Menu #1", self.show_menu)
        self.screen4 = SimpleScreen("Arbitrary Menu #2", self.show_menu)

        self.menu = MenuScreen(self.show_screen1, self.show_screen2, self.show_screen3, self.show_screen4)

        self.stack.addWidget(self.menu)    
        self.stack.addWidget(self.screen1) 
        self.stack.addWidget(self.screen2)  
        self.stack.addWidget(self.screen3)  
        self.stack.addWidget(self.screen4)  

        self.show_menu()

    def show_menu(self):
        self.stack.setCurrentWidget(self.menu)
        self.setWindowTitle("Main Menu")

    def show_screen1(self):
        self.stack.setCurrentWidget(self.screen1)
        self.setWindowTitle("Keyring Sizing Menu")

    def show_screen2(self):
        self.stack.setCurrentWidget(self.screen2)
        self.setWindowTitle("Coin Selection Menu")

    def show_screen3(self):
        self.stack.setCurrentWidget(self.screen3)
        self.setWindowTitle("Welcome to this arbitrary menu")

    def show_screen4(self):
        self.stack.setCurrentWidget(self.screen4)
        self.setWindowTitle("Welcome to this other arbitrary menu")


if __name__ == "__main__":
    app = QApplication(sys.argv)
    w = MainWindow()
    w.show()
    sys.exit(app.exec())
