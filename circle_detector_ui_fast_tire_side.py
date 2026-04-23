import math
import sys
from typing import Optional, Tuple

import cv2
import numpy as np
from PySide6.QtCore import Qt, QSize, QRectF
from PySide6.QtGui import QColor, QImage, QPainter, QPen, QPixmap
from PySide6.QtWidgets import (
    QApplication,
    QFileDialog,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QMessageBox,
    QPushButton,
    QSlider,
    QVBoxLayout,
    QWidget,
)


ARC_SAMPLES = 180


def qimage_from_bgr(img_bgr: np.ndarray) -> QImage:
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    h, w, ch = img_rgb.shape
    return QImage(img_rgb.data, w, h, ch * w, QImage.Format.Format_RGB888).copy()


def safe_normalize(signal: np.ndarray) -> np.ndarray:
    arr = np.asarray(signal, dtype=np.float32)
    if arr.size == 0:
        return arr
    mn = float(np.min(arr))
    mx = float(np.max(arr))
    if mx - mn < 1e-6:
        return np.zeros_like(arr, dtype=np.float32)
    return ((arr - mn) / (mx - mn)).astype(np.float32)


def circular_smooth(signal: np.ndarray, window: int) -> np.ndarray:
    arr = np.asarray(signal, dtype=np.float32).ravel()
    if arr.size == 0:
        return arr
    window = int(max(3, window))
    if window % 2 == 0:
        window += 1
    radius = window // 2
    x = np.arange(-radius, radius + 1, dtype=np.float32)
    sigma = max(window / 5.0, 1.0)
    kernel = np.exp(-0.5 * (x / sigma) ** 2).astype(np.float32)
    kernel /= np.sum(kernel)
    padded = np.concatenate([arr[-radius:], arr, arr[:radius]])
    return np.convolve(padded, kernel, mode="valid").astype(np.float32)


def build_skin_mask(img_bgr: np.ndarray) -> np.ndarray:
    ycrcb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2YCrCb)
    skin = cv2.inRange(ycrcb, (0, 133, 77), (255, 180, 135))
    skin = cv2.morphologyEx(skin, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
    skin = cv2.GaussianBlur(skin, (5, 5), 0)
    return skin


def unwrap_polar_strip(
    arr: np.ndarray,
    cx: float,
    cy: float,
    r: float,
    angles: np.ndarray,
    offsets: np.ndarray,
    *,
    interpolation: int = cv2.INTER_LINEAR,
) -> np.ndarray:
    cos_a = np.cos(angles).astype(np.float32)
    sin_a = np.sin(angles).astype(np.float32)
    radii = (float(r) + offsets[:, None]).astype(np.float32)
    map_x = (float(cx) + radii * cos_a[None, :]).astype(np.float32)
    map_y = (float(cy) + radii * sin_a[None, :]).astype(np.float32)
    return cv2.remap(arr, map_x, map_y, interpolation=interpolation, borderMode=cv2.BORDER_CONSTANT)


def longest_true_arc(mask: np.ndarray) -> Optional[Tuple[int, int]]:
    mask = np.asarray(mask, dtype=bool).ravel()
    n = int(mask.size)
    if n == 0 or not np.any(mask):
        return None
    if np.all(mask):
        return (0, n - 1)

    doubled = np.concatenate([mask, mask])
    best_len = 0
    best_start = 0
    run_start = None
    for i, val in enumerate(doubled):
        if val and run_start is None:
            run_start = i
        if (not val or i == len(doubled) - 1) and run_start is not None:
            end = i - 1 if not val else i
            run_len = end - run_start + 1
            if run_start < n and run_len <= n and run_len > best_len:
                best_len = run_len
                best_start = run_start
            run_start = None
    if best_len <= 0:
        return None
    start = best_start % n
    end = (best_start + best_len - 1) % n
    return (start, end)


def estimate_tire_side_arc(
    img_bgr: np.ndarray,
    cx: float,
    cy: float,
    r: float,
) -> Optional[Tuple[float, float, np.ndarray]]:
    """
    Thin outside-annulus tire-side estimator ported from the larger script.
    Returns (start_angle_rad, end_angle_rad, tire_signal).
    """
    if r < 8:
        return None

    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    eq = cv2.createCLAHE(clipLimit=2.2, tileGridSize=(8, 8)).apply(gray)
    blur = cv2.GaussianBlur(eq, (5, 5), 0)
    gx = cv2.Sobel(blur, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(blur, cv2.CV_32F, 0, 1, ksize=3)
    gradmag = cv2.magnitude(gx, gy)
    grad_ref = float(np.percentile(gradmag, 90)) + 1e-6
    edges = (cv2.Canny(blur, 45, 140) > 0).astype(np.float32)
    skin = build_skin_mask(img_bgr).astype(np.float32) / 255.0
    gray_f = blur.astype(np.float32) / 255.0
    grad_f = np.clip(gradmag / max(grad_ref, 1.0), 0.0, 2.0) / 2.0

    angles = np.linspace(0.0, 2.0 * math.pi, ARC_SAMPLES, endpoint=False, dtype=np.float32)
    outer_rhos = np.arange(max(3.0, 0.05 * r), max(8.0, 0.22 * r) + 1.0, 1.0, dtype=np.float32)
    if outer_rhos.size < 4:
        return None

    outside_gray = unwrap_polar_strip(gray_f, cx, cy, r, angles, outer_rhos)
    outside_grad = unwrap_polar_strip(grad_f, cx, cy, r, angles, outer_rhos)
    outside_edge = unwrap_polar_strip(edges, cx, cy, r, angles, outer_rhos)
    outside_skin = unwrap_polar_strip(skin, cx, cy, r, angles, outer_rhos)

    tire_raw = np.clip(
        0.56 * (1.0 - np.mean(outside_gray, axis=0))
        + 0.22 * np.mean(outside_grad, axis=0)
        + 0.14 * np.mean(outside_edge, axis=0)
        - 0.62 * np.mean(outside_skin, axis=0),
        0.0,
        1.0,
    ).astype(np.float32)

    # Mild bottom bias kept from the larger script; it helps favor the likely tire-contact side.
    tire_prior = (0.92 + 0.08 * np.clip(np.sin(angles), 0.0, 1.0)).astype(np.float32)
    tire_signal = circular_smooth(tire_raw * tire_prior, 17)
    tire_signal = safe_normalize(tire_signal)
    if float(np.max(tire_signal)) < 0.12:
        return None

    # Turn the 1-D angular signal into one continuous "tire-like" arc.
    peak = float(np.max(tire_signal))
    arc_mask = tire_signal >= max(0.58 * peak, 0.45)
    best_arc = longest_true_arc(arc_mask)
    if best_arc is None:
        peak_idx = int(np.argmax(tire_signal))
        half_width = max(10, ARC_SAMPLES // 14)
        start_idx = (peak_idx - half_width) % ARC_SAMPLES
        end_idx = (peak_idx + half_width) % ARC_SAMPLES
    else:
        start_idx, end_idx = best_arc

    return float(angles[start_idx]), float(angles[end_idx]), tire_signal


def detect_circle_fast(
    img_bgr: np.ndarray,
    init_cx: float,
    init_cy: float,
    init_r: float,
) -> Optional[Tuple[float, float, float, float]]:
    """
    Fast circle detector:
    1. Crop ROI around user's current guess
    2. Preprocess lightly
    3. Run HoughCircles
    4. Rank candidates by closeness + light rim score
    Returns: (cx, cy, r, score)
    """
    h, w = img_bgr.shape[:2]
    init_cx = float(np.clip(init_cx, 0, w - 1))
    init_cy = float(np.clip(init_cy, 0, h - 1))
    init_r = max(8.0, float(init_r))

    pad = max(20, int(round(init_r * 1.25)))
    x1 = max(0, int(round(init_cx - init_r - pad)))
    y1 = max(0, int(round(init_cy - init_r - pad)))
    x2 = min(w, int(round(init_cx + init_r + pad)))
    y2 = min(h, int(round(init_cy + init_r + pad)))

    if (x2 - x1) < 40 or (y2 - y1) < 40:
        x1, y1, x2, y2 = 0, 0, w, h

    roi = img_bgr[y1:y2, x1:x2]
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (7, 7), 0)

    roi_h, roi_w = gray.shape[:2]
    min_r = max(8, int(round(init_r * 0.60)))
    max_r = max(min_r + 4, int(round(init_r * 1.40)))
    min_r = min(min_r, max(8, min(roi_w, roi_h) // 2 - 2))
    max_r = min(max_r, max(10, min(roi_w, roi_h) // 2 - 2))
    if max_r <= min_r:
        return None

    circles = cv2.HoughCircles(
        gray,
        cv2.HOUGH_GRADIENT,
        dp=1.2,
        minDist=max(20, int(round(init_r * 0.8))),
        param1=90,
        param2=20,
        minRadius=min_r,
        maxRadius=max_r,
    )

    if circles is None or len(circles[0]) == 0:
        return None

    blur = gray
    gx = cv2.Sobel(blur, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(blur, cv2.CV_32F, 0, 1, ksize=3)
    gradmag = cv2.magnitude(gx, gy)
    grad_ref = float(np.percentile(gradmag, 90)) + 1e-6

    angles = np.linspace(0.0, 2.0 * math.pi, 72, endpoint=False, dtype=np.float32)
    cos_a = np.cos(angles).astype(np.float32)
    sin_a = np.sin(angles).astype(np.float32)

    best = None
    best_score = -1e18

    for cand in circles[0][:3]:
        x, y, r = [float(v) for v in cand]
        gx_full = x + x1
        gy_full = y + y1

        center_err = math.hypot(gx_full - init_cx, gy_full - init_cy)
        radius_err = abs(r - init_r)

        band = max(2, int(round(0.05 * r)))
        offs = np.arange(-band, band + 1, dtype=np.float32)
        radii = r + offs[:, None]
        map_x = (x + radii * cos_a[None, :]).astype(np.float32)
        map_y = (y + radii * sin_a[None, :]).astype(np.float32)

        gm = cv2.remap(gradmag, map_x, map_y, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
        gx_s = cv2.remap(gx, map_x, map_y, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
        gy_s = cv2.remap(gy, map_x, map_y, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)

        radial = gx_s * cos_a[None, :] + gy_s * sin_a[None, :]
        align = np.abs(radial) / (gm + 1e-6)
        resp = gm * (0.30 + 0.70 * align)

        cols = np.arange(resp.shape[1])
        idx = np.argmax(resp, axis=0)
        best_resp = resp[idx, cols] / grad_ref
        best_align = align[idx, cols]
        hit = (best_resp >= 0.30) & (best_align >= 0.50)

        rim_support = float(np.mean(hit))
        rim_strength = float(np.mean(np.minimum(best_resp, 1.5)) / 1.5)
        rim_align = float(np.mean(best_align[hit])) if np.any(hit) else 0.0

        score = (
            0.42 * rim_support
            + 0.24 * rim_strength
            + 0.14 * rim_align
            - 0.12 * (center_err / max(init_r, 1e-6))
            - 0.08 * (radius_err / max(init_r, 1e-6))
        )

        if score > best_score:
            best_score = score
            best = (gx_full, gy_full, r, score)

    return best


class ImageView(QLabel):
    def __init__(self):
        super().__init__()
        self.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.setMinimumSize(820, 620)
        self.setStyleSheet("background:#1f1f1f; border:1px solid #555;")
        self._img_bgr = None
        self._circle = None
        self._tire_arc = None

    def set_image(self, img_bgr: Optional[np.ndarray]):
        self._img_bgr = None if img_bgr is None else img_bgr.copy()
        self.update_view()

    def set_circle(self, circle: Optional[Tuple[float, float, float]]):
        self._circle = circle
        self.update_view()

    def set_tire_arc(self, arc: Optional[Tuple[float, float]]):
        self._tire_arc = arc
        self.update_view()

    def sizeHint(self):
        return QSize(920, 680)

    def update_view(self):
        if self._img_bgr is None:
            self.setText("Load an image to begin")
            return

        canvas = qimage_from_bgr(self._img_bgr)
        painter = QPainter(canvas)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)

        if self._circle is not None:
            cx, cy, r = self._circle

            painter.setPen(QPen(QColor(0, 180, 255), 18))
            painter.drawEllipse(
                int(round(cx - r)),
                int(round(cy - r)),
                int(round(2 * r)),
                int(round(2 * r)),
            )

            if self._tire_arc is not None:
                start_a, end_a = self._tire_arc
                span_ccw = (end_a - start_a) % (2.0 * math.pi)
                start_deg_qt = -math.degrees(start_a)
                span_deg_qt = -math.degrees(span_ccw)

                # Draw the tire-side arc slightly outside the blue circle so it is clearly visible.
                rr = r + 10
                rect = QRectF(cx - rr, cy - rr, 2 * rr, 2 * rr)
                pen = QPen(QColor(0, 255, 120), 24)
                pen.setCapStyle(Qt.PenCapStyle.RoundCap)
                painter.setPen(pen)
                painter.drawArc(rect, int(round(start_deg_qt * 16)), int(round(span_deg_qt * 16)))

        painter.end()

        pix = QPixmap.fromImage(canvas)
        self.setPixmap(
            pix.scaled(
                self.size(),
                Qt.AspectRatioMode.KeepAspectRatio,
                Qt.TransformationMode.SmoothTransformation,
            )
        )

    def resizeEvent(self, event):
        super().resizeEvent(event)
        self.update_view()


class SliderRow(QWidget):
    def __init__(self, name: str, slider: QSlider, value_label: QLabel):
        super().__init__()
        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        text = QLabel(name)
        text.setMinimumWidth(52)
        value_label.setMinimumWidth(58)
        layout.addWidget(text)
        layout.addWidget(slider, stretch=1)
        layout.addWidget(value_label)


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Reference Object Detector")
        self.slider_scale = 10.0
        self.img_bgr: Optional[np.ndarray] = None

        central = QWidget()
        self.setCentralWidget(central)
        main_layout = QHBoxLayout(central)

        self.view = ImageView()
        main_layout.addWidget(self.view, stretch=4)

        controls = QWidget()
        controls_layout = QVBoxLayout(controls)
        main_layout.addWidget(controls, stretch=1)

        load_btn = QPushButton("Upload image")
        load_btn.clicked.connect(self.load_image)
        controls_layout.addWidget(load_btn)

        detect_btn = QPushButton("Run detection")
        detect_btn.clicked.connect(self.run_detection)
        controls_layout.addWidget(detect_btn)

        self.slider_x = self._make_slider(0, 10000, 5000)
        self.slider_y = self._make_slider(0, 10000, 5000)
        self.slider_r = self._make_slider(50, 10000, 1000)

        self.label_x = QLabel("0")
        self.label_y = QLabel("0")
        self.label_r = QLabel("0")

        controls_layout.addWidget(SliderRow("X", self.slider_x, self.label_x))
        controls_layout.addWidget(SliderRow("Y", self.slider_y, self.label_y))
        controls_layout.addWidget(SliderRow("Radius", self.slider_r, self.label_r))

        for s in (self.slider_x, self.slider_y, self.slider_r):
            s.valueChanged.connect(self.overlay_changed)

        controls_layout.addStretch(1)
        self.resize(1380, 860)

    def _make_slider(self, lo, hi, value):
        s = QSlider(Qt.Orientation.Horizontal)
        s.setRange(lo, hi)
        s.setValue(value)
        s.setSingleStep(1)
        s.setPageStep(10)
        s.setTickInterval(1)
        return s

    def current_circle(self):
        return (
            float(self.slider_x.value()) / self.slider_scale,
            float(self.slider_y.value()) / self.slider_scale,
            float(self.slider_r.value()) / self.slider_scale,
        )

    def load_image(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "Select image", "", "Images (*.png *.jpg *.jpeg *.bmp *.webp)"
        )
        if not path:
            return

        img = cv2.imread(path)
        if img is None:
            QMessageBox.warning(self, "Load failed", "Could not read that image file.")
            return

        self.img_bgr = img
        h, w = img.shape[:2]

        self.slider_x.setRange(0, int((w - 1) * self.slider_scale))
        self.slider_y.setRange(0, int((h - 1) * self.slider_scale))
        self.slider_r.setRange(
            int(5 * self.slider_scale),
            int(max(10, min(w, h) // 2) * self.slider_scale),
        )

        self.slider_x.setValue(int((w / 2) * self.slider_scale))
        self.slider_y.setValue(int((h / 2) * self.slider_scale))
        self.slider_r.setValue(int(max(10, min(w, h) // 8) * self.slider_scale))

        self.view.set_image(img)
        self.view.set_tire_arc(None)
        self.overlay_changed()

    def overlay_changed(self):
        if self.img_bgr is None:
            return

        cx, cy, r = self.current_circle()
        self.label_x.setText(f"{cx:.1f}")
        self.label_y.setText(f"{cy:.1f}")
        self.label_r.setText(f"{r:.1f}")
        self.view.set_circle((cx, cy, r))
        self.view.set_tire_arc(None)

    def run_detection(self):
        if self.img_bgr is None:
            return

        cx, cy, r = self.current_circle()
        result = detect_circle_fast(self.img_bgr, cx, cy, r)

        if result is None:
            QMessageBox.information(
                self,
                "No circle found",
                "No confident circle found. Move the sliders closer to the object and try again.",
            )
            return

        x, y, rr, _score = result
        self.slider_x.setValue(int(round(x * self.slider_scale)))
        self.slider_y.setValue(int(round(y * self.slider_scale)))
        self.slider_r.setValue(int(round(rr * self.slider_scale)))

        tire_arc = estimate_tire_side_arc(self.img_bgr, x, y, rr)
        self.view.set_tire_arc(None if tire_arc is None else (tire_arc[0], tire_arc[1]))


def main():
    app = QApplication(sys.argv)
    win = MainWindow()
    win.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
