import sys
from PySide6.QtCore import Qt, QRectF
from PySide6.QtGui import QPainter, QPen
from PySide6.QtWidgets import (QApplication, QWidget, QVBoxLayout, QHBoxLayout,QPushButton, QSlider)

"""

REQUIRED: INSTALL PySide6
Use ----> pip install PySide6

"""

def mm_to_px(screen, mm):
    px_w = screen.size().width()
    mm_w = screen.physicalSize().width()
    
    return (mm / mm_w) * px_w

class CircleCanvas(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.diameter_mm = 30
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

        diameter_px = mm_to_px(screen, self.diameter_mm)

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

class CircleWidget(QWidget):
    def __init__(self, diameter_inches: float = 1.5):
        super().__init__()

        self.diameter_mm = float(diameter_inches) * 25.4

        outer = QVBoxLayout(self)
        outer.setContentsMargins(10, 10, 10, 10)
        outer.setSpacing(8)

        self.canvas = CircleCanvas(self)
        outer.addWidget(self.canvas, stretch=1)

        controls = QHBoxLayout()
        controls.setSpacing(10)
        outer.addLayout(controls)

        self.slider = QSlider(Qt.Horizontal)
        self.slider.setMinimum(100)     # 10.0 mm
        self.slider.setMaximum(1016)    # 101.6 mm (=4.0 in)
        self.slider.setValue(int(round(self.diameter_mm * 10)))
        self.slider.valueChanged.connect(self.slider_change)
        controls.addWidget(self.slider, stretch=1)
        close_btn = QPushButton("Close")
        close_btn.clicked.connect(QApplication.instance().quit)
        controls.addWidget(close_btn)

        self.resize(700, 700)
        self.apply_diameter_from_slider()

    def slider_change(self, _value: int):
        self.apply_diameter_from_slider()

    def apply_diameter_from_slider(self):
        mm = self.slider.value() / 10.0
        inches = mm / 25.4
        self.diameter_mm = mm
        self.canvas.set_diameter_mm(mm)
        self.update_window_title(inches, mm)

    def update_window_title(self, inches: float, mm: float):
        self.setWindowTitle(
            f"Diameter: {inches:.2f} in ({mm:.1f} mm)")


if __name__ == "__main__":
    app = QApplication(sys.argv)
    w = CircleWidget(diameter_inches=1.5)
    w.show()
    sys.exit(app.exec())
