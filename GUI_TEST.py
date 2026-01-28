import sys
from PySide6.QtCore import Qt, QRectF
from PySide6.QtGui import QPainter, QPen
from PySide6.QtWidgets import QApplication, QWidget, QVBoxLayout, QLabel

# Function finds display size and converts real life millimeters to pixel count
def mm_to_px_x(screen, mm: float) -> float:
    px_w = screen.size().width()
    mm_w = screen.physicalSize().width()
    return (mm/mm_w) * px_w

# Create the widget
class CircleWidget(QWidget):
    def __init__(self, diameter_mm: float = 30.0):
        super().__init__()
        self.diameter_mm = diameter_mm
        self.setWindowTitle("Circle")

        layout = QVBoxLayout(self)
        self.info = QLabel(f"{(diameter_mm/25.4):.2f} in / {diameter_mm:.2f} mm") # Display circle diameter
        self.info.setAlignment(Qt.AlignBottom)
        layout.addWidget(self.info)

        self.resize(700, 700) # Widget size

    # Draw the circle
    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing, True)
        screen = self.screen()
        
        diameter_px = mm_to_px_x(screen, self.diameter_mm) # Convert diameter in mm to pixels

        max_d = min(self.width(), self.height()) - 60 # Keep it centered in the widget
        d = min(diameter_px, max_d)

        cx = self.width()/2 # Determine center x and y
        cy = self.height()/2
        r = d/2

        rect = QRectF(cx - r, cy - r, d, d) # Make bounding rectangle with same dimensions as circle and center it

        pen = QPen(Qt.black, 3)
        painter.setPen(pen)
        painter.setBrush(Qt.NoBrush)
        painter.drawEllipse(rect) # Draw the circle
        painter.drawLine(int(cx - r), int(cy), int(cx + r), int(cy)) # Draw a horizontal diameter line

if __name__ == "__main__":
    circle_diameter_inches = 3
    circle_diameter_mm = circle_diameter_inches * 25.4
    app = QApplication(sys.argv)
    w = CircleWidget(diameter_mm=circle_diameter_mm)
    w.show()
    sys.exit(app.exec())
