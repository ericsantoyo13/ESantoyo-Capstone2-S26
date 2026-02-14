'''
REQUIRED: DOWNLOAD PYSIDE6
Use ---> pip install PySide6

Created by Eric Santoyo
'''


import sys
from PySide6.QtCore import Qt, QRectF
from PySide6.QtGui import QPainter, QPen
from PySide6.QtWidgets import (QApplication, QWidget, QVBoxLayout, QHBoxLayout, QPushButton, QSlider, QStackedWidget, QLabel)


def mm_to_px_x(screen, mm: float) -> float:
    px_w = screen.size().width()
    mm_w = screen.physicalSize().width()
    if mm_w <= 0:
        dpi = screen.logicalDotsPerInchX()
        return (mm / 25.4) * dpi
    return (mm / mm_w) * px_w


class CircleCanvas(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.diameter_mm = 30.0
        self.margin = 30

    def set_diameter_mm(self, mm: float):
        self.diameter_mm = float(mm)
        self.update()

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing, True)

        screen = self.screen()
        if not screen:
            return

        diameter_px = mm_to_px_x(screen, self.diameter_mm)

        max_d = min(self.width(), self.height()) - 2 * self.margin
        d = min(diameter_px, max_d)

        cx = self.width() / 2
        cy = self.height() / 2
        r = d / 2

        rect = QRectF(cx - r, cy - r, d, d)

        pen = QPen(Qt.black, 3)
        painter.setPen(pen)
        painter.setBrush(Qt.NoBrush)

        painter.drawEllipse(rect)
        painter.drawLine(int(cx - r), int(cy), int(cx + r), int(cy))


class SizeCircleScreen(QWidget):
    def __init__(self, go_menu_callback, diameter_inches: float = 1.5):
        super().__init__()
        self.go_menu_callback = go_menu_callback

        self.diameter_mm = float(diameter_inches) * 25.4

        outer = QVBoxLayout(self)
        outer.setContentsMargins(10, 10, 10, 10)
        outer.setSpacing(8)

        self.canvas = CircleCanvas(self)
        outer.addWidget(self.canvas, stretch=1)
        self.readout = QLabel("")
        self.readout.setAlignment(Qt.AlignCenter)
        outer.addWidget(self.readout)

        controls = QHBoxLayout()
        controls.setSpacing(10)
        outer.addLayout(controls)

        self.slider = QSlider(Qt.Horizontal)
        self.slider.setToolTip("Adjust circle diameter (mm)")
        self.slider.setMinimum(100)     # 10.0 mm
        self.slider.setMaximum(1016)    # 101.6 mm (4.00 in)
        self.slider.setValue(int(round(self.diameter_mm * 10)))
        self.slider.valueChanged.connect(self.on_slider_changed)
        controls.addWidget(self.slider, stretch=1)

        back_btn = QPushButton("Return to Menu")
        back_btn.clicked.connect(self.go_menu_callback)
        controls.addWidget(back_btn)

        close_btn = QPushButton("Exit Program")
        close_btn.clicked.connect(QApplication.instance().quit)
        controls.addWidget(close_btn)

        self.apply_diameter_from_slider()

    def on_slider_changed(self, _value: int):
        self.apply_diameter_from_slider()

    def apply_diameter_from_slider(self):
        mm = self.slider.value() / 10.0
        inches = mm / 25.4
        self.diameter_mm = mm
        self.canvas.set_diameter_mm(mm)
        self.readout.setText(f"Diameter: {inches:.2f} in ({mm:.2f} mm)")


class MenuScreen(QWidget):
    def __init__(self, go_a_callback, go_b_callback):
        super().__init__()
        layout = QVBoxLayout(self)
        layout.setContentsMargins(40, 40, 40, 40)
        layout.setSpacing(16)

        title = QLabel("What would you like to use as the reference item?")
        title.setAlignment(Qt.AlignCenter)
        layout.addWidget(title)

        btn_a = QPushButton("Keyring")
        btn_a.setMinimumHeight(44)
        btn_a.clicked.connect(go_a_callback)
        layout.addWidget(btn_a)

        btn_b = QPushButton("U.S. Coin")
        btn_b.setMinimumHeight(44)
        btn_b.clicked.connect(go_b_callback)
        layout.addWidget(btn_b)

        close_btn = QPushButton("Exit Program")
        close_btn.setMinimumHeight(44)
        close_btn.clicked.connect(QApplication.instance().quit)
        layout.addWidget(close_btn)

        layout.addStretch(1)


class CoinSelectScreen(QWidget):
    def __init__(self, go_menu_callback):
        super().__init__()
        self.go_menu_callback = go_menu_callback

        layout = QVBoxLayout(self)
        layout.setContentsMargins(40, 40, 40, 40)
        layout.setSpacing(12)

        title = QLabel("Select the type of coin you would like to use as reference")
        title.setAlignment(Qt.AlignCenter)
        layout.addWidget(title)

        self.coin_dict_mm = {"Penny": 19.05, "Nickel": 21.21, "Dime": 17.91, "Quarter": 24.26}

        for name in ["Penny", "Nickel", "Dime", "Quarter"]:
            btn = QPushButton(name)
            btn.setMinimumHeight(44)
            btn.clicked.connect(lambda _=False, n=name: self.on_coin_pressed(n))
            layout.addWidget(btn)

        self.selection_label = QLabel("Selected: (none)")
        self.selection_label.setAlignment(Qt.AlignCenter)
        layout.addWidget(self.selection_label)

        row = QHBoxLayout()
        layout.addLayout(row)

        back_btn = QPushButton("Return to Menu")
        back_btn.setMinimumHeight(44)
        back_btn.clicked.connect(self.go_menu_callback)
        row.addWidget(back_btn)

        close_btn = QPushButton("Exit Program")
        close_btn.setMinimumHeight(44)
        close_btn.clicked.connect(QApplication.instance().quit)
        row.addWidget(close_btn)

        layout.addStretch(1)

        self.selected_coin_diameter_mm = None
        self.selected_coin_name = None

    def on_coin_pressed(self, name: str):
        mm = self.coin_dict_mm[name]
        self.selected_coin_name = name
        self.selected_coin_diameter_mm = mm
        self.selection_label.setText(f"Selected: {name} ({mm:.2f} mm)")


class MainWindow(QWidget):
    def __init__(self):
        super().__init__()
        self.resize(700, 700)

        self.stack = QStackedWidget()
        root = QVBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)
        root.addWidget(self.stack)
        self.menu = MenuScreen(self.show_option_a, self.show_option_b)
        self.option_a = SizeCircleScreen(self.show_menu, diameter_inches=1.5)
        self.option_b = CoinSelectScreen(self.show_menu)

        self.stack.addWidget(self.menu)      # index 0
        self.stack.addWidget(self.option_a)  # index 1
        self.stack.addWidget(self.option_b)  # index 2

        self.show_menu()

    def show_menu(self):
        self.stack.setCurrentWidget(self.menu)
        self.setWindowTitle("Main Menu")

    def show_option_a(self):
        self.stack.setCurrentWidget(self.option_a)
        self.setWindowTitle("Circle Sizing")

    def show_option_b(self):
        self.stack.setCurrentWidget(self.option_b)
        self.setWindowTitle("Coin Reference")


if __name__ == "__main__":
    app = QApplication(sys.argv)
    w = MainWindow()
    w.show()
    sys.exit(app.exec())
