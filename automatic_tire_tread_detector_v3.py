import math
import sys
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
cv = cv2
from PySide6.QtCore import Qt, QSize, Signal, QRectF
from PySide6.QtGui import QColor, QImage, QPainter, QPen, QPixmap
from PySide6.QtWidgets import (
    QApplication,
    QButtonGroup,
    QComboBox,
    QDialog,
    QFileDialog,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QMessageBox,
    QPushButton,
    QRadioButton,
    QSlider,
    QVBoxLayout,
    QWidget,
)

COIN_DIAMETERS_MM = {"Quarter": 24.26, "Nickel": 21.21, "Penny": 19.05, "Dime": 17.91}
MANUAL_SECTOR_HALF_DEG = {"Quarter": 52.0, "Nickel": 58.0, "Penny": 64.0, "Dime": 70.0}


@dataclass
class CircleSearchConfig:
    sample_count: int = 360
    occlusion_band: int = 5
    occlusion_smooth_window: int = 17
    sector_outward_outer_frac: float = 0.30
    lbp_bins: int = 16
    chord_ransac_iterations: int = 140
    chord_angle_search_deg: float = 34.0
    chord_angle_step_deg: float = 2.5
    chord_angle_refine_tol_deg: float = 10.0
    chord_min_points: int = 8
    chord_accept_min_point_coverage: float = 0.34
    chord_depth_min_frac: float = 0.02
    chord_depth_max_frac: float = 0.72
    occlusion_retake_confidence: float = 0.42
    occlusion_hard_fail_confidence: float = 0.24
    occlusion_min_transition_confidence: float = 0.18
    reference_diameter_mm: float = 24.26
    reference_mode: str = "coin"
    forced_occlusion_angle_deg: Optional[float] = None
    forced_sector_half_deg: float = 52.0


def qimage_from_bgr(img_bgr: np.ndarray) -> QImage:
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    h, w, ch = img_rgb.shape
    return QImage(img_rgb.data, w, h, ch * w, QImage.Format.Format_RGB888).copy()


def resize_for_processing(img_bgr: np.ndarray, max_dim: int = 1400) -> Tuple[np.ndarray, float]:
    h, w = img_bgr.shape[:2]
    scale = min(1.0, float(max_dim) / max(h, w))
    if scale < 1.0:
        new_size = (int(round(w * scale)), int(round(h * scale)))
        return cv2.resize(img_bgr, new_size, interpolation=cv2.INTER_AREA), scale
    return img_bgr.copy(), 1.0


def compute_lbp_u8(gray: np.ndarray) -> np.ndarray:
    gray = np.asarray(gray, dtype=np.uint8)
    h, w = gray.shape[:2]
    if h < 3 or w < 3:
        return np.zeros_like(gray, dtype=np.uint8)

    lbp = np.zeros_like(gray, dtype=np.uint8)
    c = gray[1:-1, 1:-1]
    out = np.zeros_like(c, dtype=np.uint8)
    neighbors = [
        ((slice(0, -2), slice(0, -2)), 1),
        ((slice(0, -2), slice(1, -1)), 2),
        ((slice(0, -2), slice(2, None)), 4),
        ((slice(1, -1), slice(2, None)), 8),
        ((slice(2, None), slice(2, None)), 16),
        ((slice(2, None), slice(1, -1)), 32),
        ((slice(2, None), slice(0, -2)), 64),
        ((slice(1, -1), slice(0, -2)), 128),
    ]
    for (ys, xs), bit in neighbors:
        out |= ((gray[ys, xs] >= c).astype(np.uint8) * np.uint8(bit))
    lbp[1:-1, 1:-1] = out
    return lbp

def preprocess_for_occlusion(img_bgr: np.ndarray) -> Dict[str, np.ndarray]:
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=2.2, tileGridSize=(8, 8))
    eq = clahe.apply(gray)
    blur = cv2.GaussianBlur(eq, (5, 5), 0)
    edges = cv2.Canny(blur, 45, 140)
    edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))
    gx = cv2.Sobel(blur, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(blur, cv2.CV_32F, 0, 1, ksize=3)
    gradmag = cv2.magnitude(gx, gy)
    lbp = compute_lbp_u8(eq)
    return {"eq": eq, "edges": edges, "gx": gx, "gy": gy, "gradmag": gradmag, "lbp": lbp}


def make_occlusion_context(img_bgr: np.ndarray) -> dict:
    proc_img, proc_scale = resize_for_processing(img_bgr, max_dim=1400)
    prep = preprocess_for_occlusion(proc_img)
    grad_ref = float(np.percentile(prep["gradmag"], 90)) + 1e-6
    return {"proc_img": proc_img, "proc_scale": proc_scale, "prep": prep, "grad_ref": grad_ref}


def detect_circle_fast(
    img_bgr: np.ndarray,
    init_cx: float,
    init_cy: float,
    init_r: float,
) -> Optional[Tuple[float, float, float, float]]:
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

    gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
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


def safe_normalize(signal: np.ndarray) -> np.ndarray:
    arr = np.asarray(signal, dtype=np.float32)
    if arr.size == 0:
        return arr
    mn = float(np.min(arr))
    mx = float(np.max(arr))
    if mx - mn < 1e-6:
        return np.zeros_like(arr, dtype=np.float32)
    return ((arr - mn) / (mx - mn)).astype(np.float32)

# Build the grayscale / contrast / edge / gradient views reused by circle search.

def build_skin_mask(img_bgr: np.ndarray) -> np.ndarray:
    ycrcb = cv.cvtColor(img_bgr, cv.COLOR_BGR2YCrCb)
    skin = cv.inRange(ycrcb, (0, 133, 77), (255, 180, 135))
    skin = cv.morphologyEx(skin, cv.MORPH_OPEN, np.ones((5, 5), np.uint8))
    skin = cv.GaussianBlur(skin, (5, 5), 0)
    return skin

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
    smoothed = np.convolve(padded, kernel, mode="valid")
    return smoothed.astype(np.float32)

def unwrap_polar_strip(
    arr: np.ndarray,
    cx: float,
    cy: float,
    r: float,
    angles: np.ndarray,
    offsets: np.ndarray,
    *,
    interpolation: int = cv.INTER_LINEAR,
) -> np.ndarray:
    cos_a = np.cos(angles).astype(np.float32)
    sin_a = np.sin(angles).astype(np.float32)
    radii = (float(r) + offsets[:, None]).astype(np.float32)
    map_x = (float(cx) + radii * cos_a[None, :]).astype(np.float32)
    map_y = (float(cy) + radii * sin_a[None, :]).astype(np.float32)
    return cv.remap(arr, map_x, map_y, interpolation=interpolation, borderMode=cv.BORDER_CONSTANT)

def circle_segment_area(radius: float, height: float) -> float:
    r = float(max(radius, 1e-9))
    h = float(np.clip(height, 0.0, 2.0 * r))
    if h <= 0.0:
        return 0.0
    if h >= 2.0 * r:
        return float(math.pi * r * r)
    x = float(r - h)
    return float(r * r * math.acos(np.clip(x / r, -1.0, 1.0)) - x * math.sqrt(max(0.0, 2.0 * r * h - h * h)))

def angle_diff_rad(a: float, b: float) -> float:
    return abs((float(a) - float(b) + math.pi) % (2.0 * math.pi) - math.pi)

def weighted_fit_line_tls(points: np.ndarray, weights: np.ndarray) -> Optional[Dict[str, np.ndarray]]:
    pts = np.asarray(points, dtype=np.float64)
    if pts.ndim != 2 or pts.shape[0] < 2:
        return None
    w = np.asarray(weights, dtype=np.float64).ravel()
    if w.size != pts.shape[0]:
        w = np.ones((pts.shape[0],), dtype=np.float64)
    w = np.clip(w, 1e-6, None)
    sw = float(np.sum(w))
    if sw <= 1e-9:
        return None
    centroid = np.sum(pts * w[:, None], axis=0) / sw
    diff = pts - centroid[None, :]
    cov = (diff * w[:, None]).T @ diff / sw
    try:
        eigvals, eigvecs = np.linalg.eigh(cov)
    except np.linalg.LinAlgError:
        return None
    direction = eigvecs[:, int(np.argmax(eigvals))]
    norm = float(np.hypot(direction[0], direction[1]))
    if norm < 1e-9:
        return None
    direction = direction / norm
    return {
        'centroid': centroid.astype(np.float64),
        'direction': direction.astype(np.float64),
    }

# Robustly fit a chord/line through noisy boundary points while respecting the expected side direction.

def robust_line_fit_ransac(
    points: np.ndarray,
    weights: np.ndarray,
    *,
    cx: float,
    cy: float,
    peak_angle: float,
    angle_tol_deg: float,
    inlier_thresh: float,
    iterations: int,
) -> Optional[Dict[str, object]]:
    pts = np.asarray(points, dtype=np.float64)
    w = np.asarray(weights, dtype=np.float64).ravel()
    if pts.ndim != 2 or pts.shape[0] < 2:
        return None
    if w.size != pts.shape[0]:
        w = np.ones((pts.shape[0],), dtype=np.float64)
    w = np.clip(w, 1e-6, None)
    prob = w / np.sum(w)
    tol = math.radians(float(max(1.0, angle_tol_deg)))
    thresh = float(max(1.0, inlier_thresh))
    rng = np.random.default_rng(1337)
    best = None

    for _ in range(int(max(16, iterations))):
        try:
            idx = rng.choice(pts.shape[0], size=2, replace=False, p=prob)
        except ValueError:
            idx = rng.choice(pts.shape[0], size=2, replace=False)
        p0 = pts[int(idx[0])]
        p1 = pts[int(idx[1])]
        direction = p1 - p0
        dn = float(np.hypot(direction[0], direction[1]))
        if dn < 1e-6:
            continue
        direction = direction / dn
        nx = -direction[1]
        ny = direction[0]
        refx = math.cos(float(peak_angle))
        refy = math.sin(float(peak_angle))
        if nx * refx + ny * refy < 0.0:
            nx, ny = -nx, -ny
        phi = float(math.atan2(ny, nx))
        dev = angle_diff_rad(phi, peak_angle)
        if dev > tol:
            continue
        d = (pts[:, 0] - float(cx)) * nx + (pts[:, 1] - float(cy)) * ny
        line_d = (float(p0[0]) - float(cx)) * nx + (float(p0[1]) - float(cy)) * ny
        residual = np.abs(d - line_d)
        inliers = residual <= thresh
        if int(np.count_nonzero(inliers)) < 2:
            continue
        score = float(np.sum(w[inliers])) * max(0.0, 1.0 - dev / max(tol, 1e-6))
        if best is None or score > best['score']:
            best = {
                'score': score,
                'inliers': inliers,
                'phi_seed': phi,
                'line_d_seed': float(line_d),
            }

    if best is None:
        fit = weighted_fit_line_tls(pts, w)
        if fit is None:
            return None
        direction = fit['direction']
        nx = -direction[1]
        ny = direction[0]
        refx = math.cos(float(peak_angle))
        refy = math.sin(float(peak_angle))
        if nx * refx + ny * refy < 0.0:
            nx, ny = -nx, -ny
            direction = -direction
        phi = float(math.atan2(ny, nx))
        if angle_diff_rad(phi, peak_angle) > tol:
            return None
        d = (pts[:, 0] - float(cx)) * nx + (pts[:, 1] - float(cy)) * ny
        line_d = float(np.sum(d * w) / np.sum(w))
        residual = np.abs(d - line_d)
        inliers = residual <= thresh
        if int(np.count_nonzero(inliers)) < 2:
            return None
    else:
        inliers = np.asarray(best['inliers'], dtype=bool)
        fit = weighted_fit_line_tls(pts[inliers], w[inliers])
        if fit is None:
            fit = weighted_fit_line_tls(pts, w)
            if fit is None:
                return None
        direction = fit['direction']
        nx = -direction[1]
        ny = direction[0]
        refx = math.cos(float(peak_angle))
        refy = math.sin(float(peak_angle))
        if nx * refx + ny * refy < 0.0:
            nx, ny = -nx, -ny
            direction = -direction
        phi = float(math.atan2(ny, nx))
        if angle_diff_rad(phi, peak_angle) > tol:
            return None
        d_all = (pts[:, 0] - float(cx)) * nx + (pts[:, 1] - float(cy)) * ny
        line_d = float(np.sum(d_all[inliers] * w[inliers]) / max(np.sum(w[inliers]), 1e-9))
        residual = np.abs(d_all - line_d)
        inliers = residual <= thresh

    fit_score = float(np.sum(w[inliers]))
    return {
        'direction': np.asarray(direction, dtype=np.float64),
        'normal': np.asarray([nx, ny], dtype=np.float64),
        'phi': float(phi),
        'd': float(line_d),
        'inliers': np.asarray(inliers, dtype=bool),
        'score': fit_score,
    }

def chord_endpoints_from_normal(cx: float, cy: float, r: float, nx: float, ny: float, d: float) -> Optional[Tuple[Tuple[float, float], Tuple[float, float], Tuple[float, float]]]:
    if abs(float(d)) >= float(r):
        return None
    mx = float(cx + d * nx)
    my = float(cy + d * ny)
    tx = float(-ny)
    ty = float(nx)
    half_len = math.sqrt(max(0.0, float(r) * float(r) - float(d) * float(d)))
    p0 = (mx + half_len * tx, my + half_len * ty)
    p1 = (mx - half_len * tx, my - half_len * ty)
    mid = (mx, my)
    return p0, p1, mid

def point_sector_coverage(points: np.ndarray, cx: float, cy: float, bins: int) -> float:
    if len(points) == 0:
        return 0.0
    angles = np.arctan2(points[:, 1] - cy, points[:, 0] - cx)
    angles = (angles + 2.0 * np.pi) % (2.0 * np.pi)
    bins = int(max(4, bins))
    idx = np.floor(angles / (2.0 * np.pi) * bins).astype(int)
    idx = np.clip(idx, 0, bins - 1)
    hits = np.zeros(bins, dtype=np.float32)
    hits[np.unique(idx)] = 1.0
    return float(np.mean(hits))

def sample_ring_response(
    gradmag: np.ndarray,
    gx: np.ndarray,
    gy: np.ndarray,
    edges: np.ndarray,
    cx: float,
    cy: float,
    r: float,
    *,
    band: int,
    n_samples: int,
) -> Dict[str, np.ndarray]:
    angles = np.linspace(0.0, 2.0 * np.pi, int(n_samples), endpoint=False, dtype=np.float32)
    cos_a = np.cos(angles).astype(np.float32)
    sin_a = np.sin(angles).astype(np.float32)
    offsets = np.arange(-int(band), int(band) + 1, dtype=np.float32)
    radii = float(r) + offsets[:, None]
    map_x = (float(cx) + radii * cos_a[None, :]).astype(np.float32)
    map_y = (float(cy) + radii * sin_a[None, :]).astype(np.float32)
    gm = cv.remap(gradmag, map_x, map_y, interpolation=cv.INTER_LINEAR, borderMode=cv.BORDER_CONSTANT)
    gx_s = cv.remap(gx, map_x, map_y, interpolation=cv.INTER_LINEAR, borderMode=cv.BORDER_CONSTANT)
    gy_s = cv.remap(gy, map_x, map_y, interpolation=cv.INTER_LINEAR, borderMode=cv.BORDER_CONSTANT)
    edge_s = cv.remap(edges.astype(np.float32), map_x, map_y, interpolation=cv.INTER_LINEAR, borderMode=cv.BORDER_CONSTANT)
    radial_proj = gx_s * cos_a[None, :] + gy_s * sin_a[None, :]
    align = np.abs(radial_proj) / (gm + 1e-6)
    edge_bonus = 1.0 + 0.25 * (edge_s > 0.1).astype(np.float32)
    response = gm * (0.25 + 0.75 * align) * edge_bonus
    cols = np.arange(response.shape[1])
    best_idx = np.argmax(response, axis=0)
    best_resp = response[best_idx, cols]
    best_align = align[best_idx, cols]
    best_offset = offsets[best_idx]
    best_edge = edge_s[best_idx, cols]
    zero_i = int(np.where(offsets == 0)[0][0])
    zero_resp = response[zero_i, :]
    zero_align = align[zero_i, :]
    zero_edge = edge_s[zero_i, :]
    inward_mask = offsets <= 0
    outward_mask = offsets >= 0
    inward_resp_all = response[inward_mask, :]
    inward_align_all = align[inward_mask, :]
    inward_edge_all = edge_s[inward_mask, :]
    inward_offsets = offsets[inward_mask]
    inward_idx = np.argmax(inward_resp_all, axis=0)
    inward_best_resp = inward_resp_all[inward_idx, cols]
    inward_best_align = inward_align_all[inward_idx, cols]
    inward_best_edge = inward_edge_all[inward_idx, cols]
    inward_best_offset = inward_offsets[inward_idx]
    outward_resp_all = response[outward_mask, :]
    outward_align_all = align[outward_mask, :]
    outward_edge_all = edge_s[outward_mask, :]
    outward_offsets = offsets[outward_mask]
    outward_idx = np.argmax(outward_resp_all, axis=0)
    outward_best_resp = outward_resp_all[outward_idx, cols]
    outward_best_align = outward_align_all[outward_idx, cols]
    outward_best_edge = outward_edge_all[outward_idx, cols]
    outward_best_offset = outward_offsets[outward_idx]
    return {
        "angles": angles, "offsets": offsets.astype(np.float32),
        "best_resp": best_resp.astype(np.float32), "best_align": best_align.astype(np.float32),
        "best_offset": best_offset.astype(np.float32), "best_edge": best_edge.astype(np.float32),
        "zero_resp": zero_resp.astype(np.float32), "zero_align": zero_align.astype(np.float32),
        "zero_edge": zero_edge.astype(np.float32),
        "inward_best_resp": inward_best_resp.astype(np.float32), "inward_best_align": inward_best_align.astype(np.float32),
        "inward_best_offset": inward_best_offset.astype(np.float32), "inward_best_edge": inward_best_edge.astype(np.float32),
        "outward_best_resp": outward_best_resp.astype(np.float32), "outward_best_align": outward_best_align.astype(np.float32),
        "outward_best_offset": outward_best_offset.astype(np.float32), "outward_best_edge": outward_best_edge.astype(np.float32),
    }

def sector_surface_scores(prep: Dict[str, np.ndarray], skin_mask: np.ndarray, cx: float, cy: float, r: float, angles: np.ndarray, grad_ref: float, cfg: CircleSearchConfig) -> Dict[str, np.ndarray]:
    gray = prep["eq"]
    edges = prep["edges"].astype(np.float32)
    gradmag = prep["gradmag"].astype(np.float32)
    lbp = prep["lbp"]
    outer_start = int(max(3, round(0.04 * r)))
    outer_end = int(max(outer_start + 6, round(float(cfg.sector_outward_outer_frac) * r)))
    offsets = np.arange(outer_start, outer_end + 1, dtype=np.float32)
    if offsets.size < 6:
        z = np.zeros((angles.size,), dtype=np.float32)
        return {"tire_score": z, "skin_frac": z}
    gray_out = unwrap_polar_strip(gray, cx, cy, r, angles, offsets, interpolation=cv.INTER_LINEAR)
    edges_out = unwrap_polar_strip(edges, cx, cy, r, angles, offsets, interpolation=cv.INTER_LINEAR)
    grad_out = unwrap_polar_strip(gradmag, cx, cy, r, angles, offsets, interpolation=cv.INTER_LINEAR)
    skin_out = unwrap_polar_strip(skin_mask.astype(np.float32) / 255.0, cx, cy, r, angles, offsets, interpolation=cv.INTER_LINEAR)
    lbp_out = unwrap_polar_strip(lbp.astype(np.float32), cx, cy, r, angles, offsets, interpolation=cv.INTER_LINEAR)
    n_off = int(offsets.size)
    near_end = max(2, int(round(0.35 * n_off)))
    far_start = min(n_off - 1, max(near_end, int(round(0.60 * n_off))))
    darkness_near = np.clip(1.0 - np.mean(gray_out[:near_end, :], axis=0) / 255.0, 0.0, 1.0)
    darkness_far = np.clip(1.0 - np.mean(gray_out[far_start:, :], axis=0) / 255.0, 0.0, 1.0)
    darkness = 0.55 * darkness_near + 0.45 * darkness_far
    dark_consistency = 1.0 - np.clip(np.abs(darkness_near - darkness_far) / 0.45, 0.0, 1.0)
    edge_density = np.mean(edges_out > 0.10, axis=0).astype(np.float32)
    grad_score = (np.mean(np.clip(grad_out / max(float(grad_ref), 1.0), 0.0, 2.0), axis=0) / 2.0).astype(np.float32)
    skin_frac = np.mean(skin_out > 0.35, axis=0).astype(np.float32)
    std_n = np.clip(np.std(gray_out / 255.0, axis=0) / 0.22, 0.0, 1.0).astype(np.float32)
    lbp_bins = int(max(8, cfg.lbp_bins))
    max_entropy = math.log(float(lbp_bins)) if lbp_bins > 1 else 1.0
    ent = np.zeros((gray_out.shape[1],), dtype=np.float32)
    lbp_q = np.floor(lbp_out.astype(np.float32) * lbp_bins / 256.0).astype(np.int32)
    lbp_q = np.clip(lbp_q, 0, lbp_bins - 1)
    for col in range(lbp_q.shape[1]):
        hist = np.bincount(lbp_q[:, col], minlength=lbp_bins).astype(np.float32)
        if float(np.sum(hist)) > 0.0:
            p = hist / np.sum(hist)
            nz = p > 1e-9
            ent[col] = float(-np.sum(p[nz] * np.log(p[nz])) / max(max_entropy, 1e-6))
    texture = 0.65 * std_n + 0.35 * ent
    base = 0.26 * darkness + 0.18 * dark_consistency + 0.20 * edge_density + 0.18 * grad_score + 0.13 * texture + 0.05 * ent
    skin_gate = np.clip(1.0 - np.maximum(0.0, skin_frac - 0.05) / 0.18, 0.0, 1.0)
    tire_score = np.clip(base * skin_gate - 0.70 * skin_frac, 0.0, 1.5).astype(np.float32)
    return {"tire_score": tire_score, "skin_frac": skin_frac.astype(np.float32)}

def safe_weighted_mean(values: np.ndarray, weights: np.ndarray) -> float:
    v = np.asarray(values, dtype=np.float32).ravel()
    w = np.asarray(weights, dtype=np.float32).ravel()
    if v.size == 0 or w.size == 0 or v.size != w.size:
        return 0.0
    w = np.maximum(w, 0.0)
    s = float(np.sum(w))
    if s <= 1e-8:
        return 0.0
    return float(np.sum(v * w) / s)

def sample_points_linear(img: np.ndarray, xs: np.ndarray, ys: np.ndarray) -> np.ndarray:
    xs = np.asarray(xs, dtype=np.float32).ravel()
    ys = np.asarray(ys, dtype=np.float32).ravel()
    if xs.size == 0:
        return np.zeros((0,), dtype=np.float32)
    arr = img.astype(np.float32)
    max_chunk = 30000
    out = np.empty((xs.size,), dtype=np.float32)
    for start in range(0, xs.size, max_chunk):
        end = min(start + max_chunk, xs.size)
        map_x = xs[start:end].reshape(1, -1)
        map_y = ys[start:end].reshape(1, -1)
        sampled = cv.remap(arr, map_x, map_y, interpolation=cv.INTER_LINEAR, borderMode=cv.BORDER_REFLECT101)
        out[start:end] = sampled.reshape(-1).astype(np.float32)
    return out

def chord_endpoints_from_normal(cx: float, cy: float, r: float, nx: float, ny: float, d: float) -> Optional[Tuple[Tuple[float, float], Tuple[float, float], Tuple[float, float]]]:
    if abs(float(d)) >= float(r):
        return None
    mx = float(cx + d * nx)
    my = float(cy + d * ny)
    tx = float(-ny)
    ty = float(nx)
    half_len = math.sqrt(max(0.0, float(r) * float(r) - float(d) * float(d)))
    p0 = (mx + half_len * tx, my + half_len * ty)
    p1 = (mx - half_len * tx, my - half_len * ty)
    return p0, p1, (mx, my)

def detect_keyring_inner_circle(context: dict, final_circle_orig: dict, cfg: CircleSearchConfig) -> Optional[Dict[str, float]]:
    proc_scale = float(context["proc_scale"])
    prep = context["prep"]
    grad_ref = float(context["grad_ref"])
    cx = float(final_circle_orig["cx"] * proc_scale)
    cy = float(final_circle_orig["cy"] * proc_scale)
    r_outer = float(final_circle_orig["r"] * proc_scale)
    if r_outer < 10.0:
        return None
    radii = np.linspace(max(5.0, 0.22 * r_outer), max(6.0, 0.78 * r_outer), int(max(16, round(0.36 * r_outer))))
    best = None
    for r_in in radii.tolist():
        on = sample_ring_response(prep["gradmag"], prep["gx"], prep["gy"], prep["edges"], cx, cy, float(r_in), band=3, n_samples=max(240, cfg.sample_count))
        resp = np.clip(on["zero_resp"] / max(grad_ref, 1.0), 0.0, 2.5)
        align = np.clip(on["zero_align"], 0.0, 1.0)
        edge = (on["zero_edge"] > 0.10).astype(np.float32)
        support_mask = (resp >= 0.28) & (align >= 0.50)
        support = float(np.mean(support_mask))
        if support < 0.22:
            continue
        if np.any(support_mask):
            strength = float(np.mean(resp[support_mask]))
            align_mean = float(np.mean(align[support_mask]))
            edge_mean = float(np.mean(edge[support_mask]))
        else:
            strength = float(np.mean(resp))
            align_mean = float(np.mean(align))
            edge_mean = float(np.mean(edge))
        contrast = float(np.clip(np.mean(np.maximum(0.0, on["zero_resp"] - 0.55 * on["inward_best_resp"] - 0.40 * on["outward_best_resp"])) / max(grad_ref, 1.0), 0.0, 1.0))
        ratio = float(r_in / max(r_outer, 1e-6))
        ratio_pref = float(np.clip(1.0 - abs(ratio - 0.58) / 0.28, 0.0, 1.0))
        score = 0.36 * support + 0.26 * min(strength / 1.15, 1.0) + 0.18 * align_mean + 0.10 * edge_mean + 0.07 * contrast + 0.03 * ratio_pref
        if best is None or score > best["score"]:
            best = {"cx": cx, "cy": cy, "r": float(r_in), "score": float(score), "support": support, "ratio": ratio}
    if best is None or best["score"] < 0.34 or best["support"] < 0.28 or best["ratio"] > 0.86:
        return None
    scale_back = 1.0 / max(proc_scale, 1e-9)
    return {"cx": float(best["cx"] * scale_back), "cy": float(best["cy"] * scale_back), "r": float(best["r"] * scale_back), "score": float(best["score"])}

def _fallback_coin_like_occlusion(context: dict, final_circle_orig: dict, cfg: CircleSearchConfig) -> Dict[str, object]:
    prev_mode = getattr(cfg, "reference_mode", "coin")
    try:
        cfg.reference_mode = "coin"
        out = detect_occlusion_from_circle(context, final_circle_orig, cfg)
    finally:
        cfg.reference_mode = prev_mode
    if out.get("found"):
        out = dict(out)
        out["measurement_mode"] = f"{out.get('measurement_mode', 'boundary_points_ransac_chord')}+fallback"
    return out

def detect_occlusion_from_keyring_hollow(context: dict, final_circle_orig: dict, cfg: CircleSearchConfig) -> Dict[str, object]:
    proc_scale = float(context["proc_scale"])
    prep = context["prep"]
    proc_img = context["proc_img"]
    grad_ref = float(context["grad_ref"])
    cx = float(final_circle_orig["cx"] * proc_scale)
    cy = float(final_circle_orig["cy"] * proc_scale)
    r = float(final_circle_orig["r"] * proc_scale)
    if r < 6.0:
        return {"found": False, "reason": "circle too small"}
    band = int(max(2, cfg.occlusion_band))
    on = sample_ring_response(prep["gradmag"], prep["gx"], prep["gy"], prep["edges"], cx, cy, r, band=band, n_samples=cfg.sample_count)
    denom = max(1.0, grad_ref)
    zero_norm = np.clip(on["zero_resp"] / denom, 0.0, 2.5)
    inward_norm = np.clip(on["inward_best_resp"] / denom, 0.0, 2.5)
    outward_norm = np.clip(on["outward_best_resp"] / denom, 0.0, 2.5)
    zero_strength = np.clip(zero_norm / 1.15, 0.0, 1.0)
    inward_strength = np.clip(inward_norm / 1.15, 0.0, 1.0)
    zero_edge = (on["zero_edge"] > 0.10).astype(np.float32)
    zero_local_adv = np.clip(zero_norm - 0.60 * inward_norm - 0.35 * outward_norm, 0.0, 2.5)
    zero_visibility = 0.44 * zero_strength + 0.24 * on["zero_align"] + 0.18 * np.clip(zero_local_adv / 0.90, 0.0, 1.0) + 0.14 * zero_edge
    inward_shift_px = np.maximum(0.0, -on["inward_best_offset"])
    inward_shift = np.clip((inward_shift_px - 0.75) / max(1.0, float(band) - 0.75), 0.0, 1.0)
    inward_adv = np.clip((inward_norm - zero_norm) / 0.70, 0.0, 1.0)
    occ_s = circular_smooth((0.42 * np.clip(1.0 - zero_visibility, 0.0, 1.0) + 0.32 * inward_shift + 0.26 * inward_adv) * (0.65 + 0.35 * inward_strength), int(cfg.occlusion_smooth_window))
    skin_mask = build_skin_mask(proc_img)
    sector_info = sector_surface_scores(prep, skin_mask, cx, cy, r, on["angles"], grad_ref, cfg)
    tire_sector_s = circular_smooth(sector_info["tire_score"], max(11, int(cfg.occlusion_smooth_window) + 6))
    skin_sector_s = circular_smooth(sector_info["skin_frac"], max(9, int(cfg.occlusion_smooth_window) + 2))
    tire_sector_n = safe_normalize(tire_sector_s)
    occ_s_n = safe_normalize(occ_s)
    combined_seed = np.clip(0.18 * occ_s_n + 0.98 * tire_sector_n - 0.72 * np.clip(skin_sector_s, 0.0, 1.0), 0.0, 1.0)
    combined_seed_s = circular_smooth(combined_seed, max(15, int(cfg.occlusion_smooth_window) + 8))
    peak_idx = int(np.argmax(combined_seed_s))
    peak_val = float(combined_seed_s[peak_idx])
    inner_circle = detect_keyring_inner_circle(context, final_circle_orig, cfg)
    inner_r = float(inner_circle["r"] * proc_scale) if inner_circle is not None else float(np.clip(0.82 * r, 6.0, max(6.0, r - 2.0)))
    if peak_val < 0.08:
        fb = _fallback_coin_like_occlusion(context, final_circle_orig, cfg)
        if inner_circle is not None:
            fb = dict(fb)
            fb["inner_circle"] = inner_circle
        return fb
    peak_angle = float(on["angles"][peak_idx])
    gray_u8 = prep["eq"].astype(np.uint8)
    gray = cv.GaussianBlur(gray_u8.astype(np.float32) / 255.0, (5, 5), 0)
    edge_img = (prep["edges"].astype(np.float32) > 0.10).astype(np.float32)
    gradmag = prep["gradmag"].astype(np.float32)
    gx = prep["gx"].astype(np.float32)
    gy = prep["gy"].astype(np.float32)
    grad_n_img = (np.clip(gradmag / max(float(grad_ref), 1.0), 0.0, 2.0) / 2.0).astype(np.float32)
    lap = cv.Laplacian(gray_u8, cv.CV_32F, ksize=3)
    tex = cv.GaussianBlur(np.abs(lap), (5, 5), 0)
    tex_n = np.clip(tex / 42.0, 0.0, 1.0).astype(np.float32)
    skin_n = np.clip(skin_mask.astype(np.float32) / 255.0, 0.0, 1.0)
    yy, xx = np.mgrid[0:gray.shape[0], 0:gray.shape[1]]
    dx_full = xx.astype(np.float32) - float(cx)
    dy_full = yy.astype(np.float32) - float(cy)
    dist = np.sqrt(dx_full * dx_full + dy_full * dy_full)
    inner_mask = dist <= inner_r
    edge_mask = (edge_img > 0.5) & inner_mask & (skin_n < 0.45)
    ys, xs = np.nonzero(edge_mask)
    if xs.size < 18:
        fb = _fallback_coin_like_occlusion(context, final_circle_orig, cfg)
        if inner_circle is not None:
            fb = dict(fb)
            fb["inner_circle"] = inner_circle
        return fb
    pts = np.column_stack([xs.astype(np.float32), ys.astype(np.float32)])
    dx = dx_full[ys, xs].astype(np.float32)
    dy = dy_full[ys, xs].astype(np.float32)
    pix_gx = gx[ys, xs].astype(np.float32)
    pix_gy = gy[ys, xs].astype(np.float32)
    pix_gm = np.maximum(1e-6, gradmag[ys, xs].astype(np.float32))
    pix_tex = tex_n[ys, xs].astype(np.float32)
    pix_skin = skin_n[ys, xs].astype(np.float32)
    angle_span = float(max(10.0, min(40.0, float(cfg.chord_angle_search_deg))))
    angle_step = float(max(0.75, float(cfg.chord_angle_step_deg)))
    angle_offsets_deg = np.arange(-angle_span, angle_span + 0.5 * angle_step, angle_step, dtype=np.float32)
    line_band = float(max(2.0, 0.030 * inner_r))
    delta = float(max(1.8, 0.050 * inner_r))
    min_points = int(max(8, cfg.chord_min_points))
    hist_bins = int(max(20, round(0.50 * inner_r)))
    best = None
    candidate_count = 0
    for dphi_deg in angle_offsets_deg.tolist():
        phi = float((peak_angle + math.radians(dphi_deg)) % (2.0 * math.pi))
        nx = math.cos(phi)
        ny = math.sin(phi)
        tx = -ny
        ty = nx
        grad_align = np.clip(np.abs((pix_gx * nx + pix_gy * ny)) / pix_gm, 0.0, 1.0)
        signed = dx * nx + dy * ny
        along = dx * tx + dy * ty
        base_valid = (grad_align >= 0.55) & (signed >= -0.98 * inner_r) & (signed <= 0.98 * inner_r) & (pix_skin < 0.35)
        if int(np.count_nonzero(base_valid)) < min_points:
            continue
        base_w = np.clip(0.55 * grad_align + 0.35 * np.clip(pix_gm / max(float(grad_ref), 1.0), 0.0, 1.6) / 1.6 + 0.10 * pix_tex - 0.35 * pix_skin, 0.0, 1.0)
        hist, edges_d = np.histogram(signed[base_valid], bins=hist_bins, range=(-1.02 * inner_r, 1.02 * inner_r), weights=base_w[base_valid])
        if hist.size == 0 or float(np.max(hist)) <= 0.0:
            continue
        top_bins = np.argsort(hist)[-4:]
        top_bins = top_bins[np.argsort(hist[top_bins])[::-1]]
        for bi in top_bins.tolist():
            d_center = 0.5 * (float(edges_d[bi]) + float(edges_d[bi + 1]))
            near = base_valid & (np.abs(signed - d_center) <= line_band)
            if int(np.count_nonzero(near)) < min_points:
                continue
            coverage = float(np.clip((float(np.max(along[near])) - float(np.min(along[near]))) / max(2.0 * inner_r, 1e-6), 0.0, 1.0))
            if coverage < 0.12:
                continue
            fit = robust_line_fit_ransac(pts[near], base_w[near], cx=cx, cy=cy, peak_angle=phi, angle_tol_deg=max(5.0, 0.45 * angle_span), inlier_thresh=max(1.5, 1.25 * line_band), iterations=max(120, int(0.30 * cfg.chord_ransac_iterations)))
            if fit is None:
                continue
            nx_f = float(fit["normal"][0]); ny_f = float(fit["normal"][1]); d_f = float(fit["d"])
            endpoints = chord_endpoints_from_normal(cx, cy, inner_r, nx_f, ny_f, d_f)
            if endpoints is None:
                continue
            chord_p0, chord_p1, chord_mid = endpoints
            half_focus = math.sqrt(max(0.0, inner_r * inner_r - min(d_f * d_f, inner_r * inner_r)))
            if half_focus < 6.0:
                continue
            tx_f = -ny_f; ty_f = nx_f
            us = np.linspace(-half_focus, half_focus, 96, dtype=np.float32)
            xs_line = chord_mid[0] + us * tx_f
            ys_line = chord_mid[1] + us * ty_f
            edge_line = sample_points_linear(edge_img, xs_line, ys_line)
            grad_line = sample_points_linear(grad_n_img, xs_line, ys_line)
            tex_line = sample_points_linear(tex_n, xs_line, ys_line)
            gray_neg = sample_points_linear(gray, xs_line - delta * nx_f, ys_line - delta * ny_f)
            gray_pos = sample_points_linear(gray, xs_line + delta * nx_f, ys_line + delta * ny_f)
            tex_neg = sample_points_linear(tex_n, xs_line - delta * nx_f, ys_line - delta * ny_f)
            tex_pos = sample_points_linear(tex_n, xs_line + delta * nx_f, ys_line + delta * ny_f)
            edge_neg = sample_points_linear(edge_img, xs_line - delta * nx_f, ys_line - delta * ny_f)
            edge_pos = sample_points_linear(edge_img, xs_line + delta * nx_f, ys_line + delta * ny_f)
            skin_cross = np.maximum(sample_points_linear(skin_n, xs_line, ys_line), np.maximum(sample_points_linear(skin_n, xs_line - delta * nx_f, ys_line - delta * ny_f), sample_points_linear(skin_n, xs_line + delta * nx_f, ys_line + delta * ny_f)))
            line_support = float(np.clip(np.mean(0.52 * edge_line + 0.32 * grad_line + 0.16 * tex_line - 0.40 * skin_cross), 0.0, 1.0))
            texture_sep = float(np.clip(np.mean(tex_pos - tex_neg + 0.06) / 0.32, 0.0, 1.0))
            edge_sep = float(np.clip(np.mean(edge_pos - edge_neg + 0.03) / 0.22, 0.0, 1.0))
            gray_sep = float(np.clip(np.mean(np.abs(gray_pos - gray_neg)) / 0.18, 0.0, 1.0))
            band_sep = float(np.clip(0.44 * texture_sep + 0.28 * edge_sep + 0.28 * gray_sep, 0.0, 1.0))
            point_mean = safe_weighted_mean(base_w[near], base_w[near])
            fit_inliers = np.asarray(fit["inliers"], dtype=bool)
            if fit_inliers.size != int(np.count_nonzero(near)):
                continue
            fit_pts = pts[near][fit_inliers]
            if fit_pts.shape[0] < min_points:
                continue
            fit_dx = fit_pts[:, 0] - float(cx)
            fit_dy = fit_pts[:, 1] - float(cy)
            fit_along = fit_dx * tx_f + fit_dy * ty_f
            point_coverage = float(np.clip((float(np.max(fit_along)) - float(np.min(fit_along))) / max(2.0 * half_focus, 1e-6), 0.0, 1.0))
            point_support = float(np.clip(np.mean(base_w[near][fit_inliers]), 0.0, 1.0))
            seed_match = float(max(0.0, 1.0 - angle_diff_rad(float(fit["phi"]), peak_angle) / math.radians(max(10.0, angle_span))))
            confidence = float(np.clip(0.34 * line_support + 0.28 * band_sep + 0.18 * point_coverage + 0.12 * point_support + 0.08 * seed_match, 0.0, 1.0))
            score = float(np.clip(0.40 * line_support + 0.24 * band_sep + 0.18 * point_coverage + 0.10 * point_mean + 0.08 * seed_match, 0.0, 1.0))
            candidate_count += 1
            if best is None or score > best["score"]:
                rim_pt = (float(cx + r * nx_f), float(cy + r * ny_f))
                best = {
                    "score": score, "confidence": confidence, "depth_proc": float(max(0.0, r - d_f)), "phi": float(fit["phi"]),
                    "line_support": line_support, "band_sep": band_sep, "point_support": point_support, "point_coverage": point_coverage,
                    "point_mean": point_mean, "line": {"p0": chord_p0, "p1": chord_p1, "mid": chord_mid},
                    "measurement": {"p0": rim_pt, "p1": chord_mid, "mid": ((rim_pt[0] + chord_mid[0]) * 0.5, (rim_pt[1] + chord_mid[1]) * 0.5)},
                    "fill_area_proc": circle_segment_area(r, max(0.0, r - d_f)), "fill_fraction": float(circle_segment_area(r, max(0.0, r - d_f)) / max(math.pi * r * r, 1e-6)),
                }
    if best is None:
        fb = _fallback_coin_like_occlusion(context, final_circle_orig, cfg)
        if inner_circle is not None:
            fb = dict(fb); fb["inner_circle"] = inner_circle
        if fb.get("found"):
            return fb
        return {"found": False, "reason": "no keyring chord candidate", "candidate_count": candidate_count, "inner_circle": inner_circle}
    retake_recommended = bool(best["confidence"] < max(0.34, float(cfg.occlusion_retake_confidence) - 0.04) or best["line_support"] < max(0.14, float(cfg.occlusion_min_transition_confidence) - 0.02) or best["point_coverage"] < max(0.18, float(cfg.chord_accept_min_point_coverage) - 0.02))
    if best["confidence"] < max(0.18, float(cfg.occlusion_hard_fail_confidence) - 0.04) or best["line_support"] < 0.10 or best["point_coverage"] < 0.10:
        fb = _fallback_coin_like_occlusion(context, final_circle_orig, cfg)
        if inner_circle is not None:
            fb = dict(fb); fb["inner_circle"] = inner_circle
        if fb.get("found") and float(fb.get("confidence", 0.0)) >= float(best["confidence"]):
            return fb
        return {"found": False, "reason": "low keyring occlusion confidence", "confidence": best["confidence"], "retake_recommended": True, "candidate_count": candidate_count, "inner_circle": inner_circle}
    scale_back = 1.0 / max(proc_scale, 1e-9)
    depth_px = float(best["depth_proc"] * scale_back)
    depth_mm = float(best["depth_proc"] * float(cfg.reference_diameter_mm) / max(2.0 * r, 1e-6)) if float(cfg.reference_diameter_mm) > 0.0 else None
    chord_p0, chord_p1, chord_mid = best["line"]["p0"], best["line"]["p1"], best["line"]["mid"]
    meas_p0, meas_p1, meas_mid = best["measurement"]["p0"], best["measurement"]["p1"], best["measurement"]["mid"]
    line_samples = []
    for u in np.linspace(0.0, 1.0, 40, dtype=np.float32):
        x = (1.0 - float(u)) * chord_p0[0] + float(u) * chord_p1[0]
        y = (1.0 - float(u)) * chord_p0[1] + float(u) * chord_p1[1]
        line_samples.append({"x": float(x * scale_back), "y": float(y * scale_back)})
    out = {
        "found": True, "score": float(best["score"]), "confidence": float(best["confidence"]), "retake_recommended": retake_recommended,
        "confidence_reason": "low confidence" if retake_recommended else "ok", "transition_mean": float(best["line_support"]),
        "path_consistency": float(best["point_support"]), "fit_agreement": float(best["point_coverage"]), "fit_blend": 1.0,
        "lowness": float(np.clip(best["band_sep"], 0.0, 1.0)), "transition": float(best["line_support"]), "span_deg": 0.0,
        "start_angle_deg": float((math.degrees(best["phi"]) - 90.0) % 360.0), "end_angle_deg": float((math.degrees(best["phi"]) + 90.0) % 360.0),
        "mid_angle_deg": float(math.degrees(best["phi"]) % 360.0), "depth_px": depth_px, "depth_mm": depth_mm,
        "depth_frac": float(best["depth_proc"] / max(r, 1e-6)), "threshold": float(best["score"]), "measurement_mode": "keyring_hollow_edge_chord",
        "measurement": {"p0": {"x": float(meas_p0[0] * scale_back), "y": float(meas_p0[1] * scale_back)}, "p1": {"x": float(meas_p1[0] * scale_back), "y": float(meas_p1[1] * scale_back)}, "mid": {"x": float(meas_mid[0] * scale_back), "y": float(meas_mid[1] * scale_back)}, "angle_deg": float(math.degrees(best["phi"]) % 360.0)},
        "chord": {"p0": {"x": float(chord_p0[0] * scale_back), "y": float(chord_p0[1] * scale_back)}, "p1": {"x": float(chord_p1[0] * scale_back), "y": float(chord_p1[1] * scale_back)}, "mid": {"x": float(chord_mid[0] * scale_back), "y": float(chord_mid[1] * scale_back)}},
        "boundary_polyline": line_samples, "fill_fraction": float(best["fill_fraction"]), "fill_area_px2": float(best["fill_area_proc"] * scale_back * scale_back),
        "robust_fill_depth_px": depth_px, "mean_fill_depth_px": depth_px, "peak_fill_depth_px": depth_px, "chord_depth_px": depth_px, "chord_depth_mm": depth_mm,
        "candidate_count": int(candidate_count), "path_score": float(best["score"]), "line_support": float(best["line_support"]), "band_separation": float(best["band_sep"]),
        "region_separation": float(best["band_sep"]), "point_support": float(best["point_support"]), "point_coverage": float(best["point_coverage"]), "point_mean": float(best["point_mean"]),
        "inner_circle": inner_circle,
    }
    return out

# Convert the ROI fractions into concrete pixel-space search bounds and cached preprocessing results.

def detect_occlusion_from_circle(context: dict, final_circle_orig: dict, cfg: CircleSearchConfig) -> Dict[str, object]:
    if getattr(cfg, "reference_mode", "coin") == "keyring":
        return detect_occlusion_from_keyring_hollow(context, final_circle_orig, cfg)

    proc_scale = float(context["proc_scale"])
    prep = context["prep"]
    proc_img = context["proc_img"]
    grad_ref = float(context["grad_ref"])

    cx = float(final_circle_orig["cx"] * proc_scale)
    cy = float(final_circle_orig["cy"] * proc_scale)
    r = float(final_circle_orig["r"] * proc_scale)

    if r < 6.0:
        return {"found": False, "reason": "circle too small"}

    # Scale-adaptive tuning.
    size_alpha = float(np.clip((r - 24.0) / 34.0, 0.0, 1.0))  # 0=small in image, 1=large
    smallness = float(1.0 - size_alpha)

    blur_k = int(5 + 4 * size_alpha)
    if blur_k % 2 == 0:
        blur_k += 1

    gray = prep["eq"].astype(np.float32) / 255.0
    gray_scan = cv.GaussianBlur(prep["eq"], (blur_k, blur_k), 0).astype(np.float32) / 255.0
    grad = np.clip(prep["gradmag"].astype(np.float32) / max(grad_ref, 1.0), 0.0, 2.0) / 2.0
    edges = (prep["edges"].astype(np.float32) > 0.0).astype(np.float32)
    skin = build_skin_mask(proc_img).astype(np.float32) / 255.0

    n_angles = int(max(360, cfg.sample_count * 2))
    angles = np.linspace(0.0, 2.0 * np.pi, n_angles, endpoint=False, dtype=np.float32)

    def sample_ray(arr: np.ndarray, theta: float, rhos: np.ndarray) -> np.ndarray:
        map_x = (cx + rhos * math.cos(float(theta))).astype(np.float32).reshape(1, -1)
        map_y = (cy + rhos * math.sin(float(theta))).astype(np.float32).reshape(1, -1)
        vals = cv.remap(arr, map_x, map_y, interpolation=cv.INTER_LINEAR, borderMode=cv.BORDER_REFLECT101)
        return vals.reshape(-1).astype(np.float32)

    # Step 1: estimate the tire-contact side from a thin outside annulus.
    outer_rhos = np.arange(max(3.0, 0.05 * r), max(8.0, 0.22 * r) + 1.0, 1.0, dtype=np.float32)
    if outer_rhos.size < 4:
        return {"found": False, "reason": "outer annulus too small"}

    outside_gray = unwrap_polar_strip(gray, cx, cy, r, angles, outer_rhos, interpolation=cv.INTER_LINEAR)
    outside_grad = unwrap_polar_strip(grad, cx, cy, r, angles, outer_rhos, interpolation=cv.INTER_LINEAR)
    outside_edge = unwrap_polar_strip(edges, cx, cy, r, angles, outer_rhos, interpolation=cv.INTER_LINEAR)
    outside_skin = unwrap_polar_strip(skin, cx, cy, r, angles, outer_rhos, interpolation=cv.INTER_LINEAR)

    tire_raw = np.clip(
        0.56 * (1.0 - np.mean(outside_gray, axis=0))
        + 0.22 * np.mean(outside_grad, axis=0)
        + 0.14 * np.mean(outside_edge, axis=0)
        - 0.62 * np.mean(outside_skin, axis=0),
        0.0,
        1.0,
    ).astype(np.float32)
    tire_prior = (0.92 + 0.08 * np.clip(np.sin(angles), 0.0, 1.0)).astype(np.float32)
    tire_signal = circular_smooth(tire_raw * tire_prior, max(13, int(cfg.occlusion_smooth_window)))
    tire_signal = safe_normalize(tire_signal)

    forced_angle_deg = getattr(cfg, "forced_occlusion_angle_deg", None)
    if forced_angle_deg is not None:
        peak_angle = math.radians(float(forced_angle_deg) % 360.0)
        peak_val = 1.0
    else:
        peak_idx = int(np.argmax(tire_signal))
        peak_angle = float(angles[peak_idx])
        peak_val = float(tire_signal[peak_idx])
        if peak_val < 0.12:
            return {"found": False, "reason": "weak tire-side signal"}

    # Step 2: radial scanning inside the coin.
    if forced_angle_deg is not None:
        sector_half_deg = float(getattr(cfg, "forced_sector_half_deg", 52.0))
    else:
        sector_half_deg = float(66.0 - 10.0 * size_alpha)  # wider for small coins
    sector_half_rad = math.radians(sector_half_deg)
    angle_diffs = np.abs((angles - peak_angle + math.pi) % (2.0 * math.pi) - math.pi)
    sector_idxs = np.where(angle_diffs <= sector_half_rad)[0]
    if sector_idxs.size < 16:
        return {"found": False, "reason": "scan sector too small"}

    n_rays = int(min(104, max(44, sector_idxs.size)))
    if sector_idxs.size > n_rays:
        choose = np.linspace(0, sector_idxs.size - 1, n_rays).round().astype(int)
        sector_idxs = sector_idxs[choose]

    # Smaller detections need much more inward reach.
    inner_start_frac = float(0.14 + 0.26 * size_alpha)
    inner_stop_frac = float(0.96 + 0.03 * size_alpha)
    inner_start = float(max(inner_start_frac * r, 3.0))
    inner_stop = float(max(inner_start + 8.0, inner_stop_frac * r))
    ray_count = int(max(48, round((0.90 - 0.08 * size_alpha) * r)))
    ray_rhos = np.linspace(inner_start, inner_stop, ray_count, dtype=np.float32)
    if ray_rhos.size < 18:
        return {"found": False, "reason": "ray profile too short"}

    inside_w = max(2, int(round((0.028 + 0.012 * size_alpha) * ray_rhos.size)))
    outside_w = max(3, int(round((0.040 + 0.015 * size_alpha) * ray_rhos.size)))
    min_idx = max(inside_w + 1, int(round((0.08 + 0.14 * size_alpha) * ray_rhos.size)))
    max_idx = min(len(ray_rhos) - outside_w - 4, int(round((0.95 + 0.03 * size_alpha) * ray_rhos.size)))
    if max_idx <= min_idx + 2:
        return {"found": False, "reason": "invalid ray search bounds"}

    contrast_min = float(0.038 + 0.010 * size_alpha)
    tail_dark_min = float(0.036 + 0.010 * size_alpha)
    local_drop_min = float(0.022 + 0.006 * size_alpha)
    persistent_dark_min = float(0.46 + 0.08 * size_alpha)
    best_score_min = float(0.20 + 0.05 * size_alpha)
    tail_len_frac = float(0.11 + 0.05 * size_alpha)

    candidate_pts = []
    candidate_w = []
    candidate_angles = []
    candidate_rhos = []
    candidate_rel = []
    candidate_debug = []

    for idx in sector_idxs:
        th = float(angles[int(idx)])
        gray_prof = sample_ray(gray_scan, th, ray_rhos)
        grad_prof = sample_ray(grad, th, ray_rhos)
        edge_prof = sample_ray(edges, th, ray_rhos)
        skin_prof = sample_ray(skin, th, ray_rhos)

        outside_probe_rhos = np.linspace(1.02 * r, 1.18 * r, 8, dtype=np.float32)
        outside_probe = sample_ray(gray_scan, th, outside_probe_rhos)
        outside_dark = float(np.clip(1.0 - np.mean(outside_probe), 0.0, 1.0))
        seed_match = float(max(0.0, 1.0 - angle_diffs[int(idx)] / max(sector_half_rad, 1e-6)))

        best_score = -1e9
        best_i = None
        for i in range(min_idx, max_idx + 1):
            inside_slice = gray_prof[max(0, i - inside_w):i]
            outside_slice = gray_prof[i + 1:min(len(gray_prof), i + 1 + outside_w)]
            tail_slice = gray_prof[i + 1:]
            tail_grad_slice = grad_prof[i + 1:]
            tail_edge_slice = edge_prof[i + 1:]
            if inside_slice.size < 2 or outside_slice.size < 2 or tail_slice.size < 6:
                continue

            inside_mean = float(np.mean(inside_slice))
            outside_mean = float(np.mean(outside_slice))
            contrast = max(0.0, inside_mean - outside_mean)

            pre_slice = gray_prof[max(0, i - 2):i]
            post_slice = gray_prof[i + 1:min(len(gray_prof), i + 3)]
            if pre_slice.size < 1 or post_slice.size < 1:
                continue
            local_drop = max(0.0, float(np.mean(pre_slice) - np.mean(post_slice)))

            tail_len = max(outside_w + 2, int(round(tail_len_frac * len(ray_rhos))))
            tail_probe = gray_prof[i + 1:min(len(gray_prof), i + 1 + tail_len)]
            tail_probe_grad = grad_prof[i + 1:min(len(grad_prof), i + 1 + tail_len)]
            tail_probe_edge = edge_prof[i + 1:min(len(edge_prof), i + 1 + tail_len)]
            if tail_probe.size < 4:
                continue

            tail_mean = float(np.mean(tail_probe))
            tail_dark = max(0.0, inside_mean - tail_mean)
            persistent_dark_frac = float(np.mean(tail_slice <= (inside_mean - (0.034 + 0.005 * size_alpha))))
            persistent_tire_score = float(np.clip(
                0.72 * (1.0 - np.mean(tail_probe))
                + 0.18 * np.mean(tail_probe_grad)
                + 0.10 * np.mean(tail_probe_edge),
                0.0,
                1.0,
            ))
            rebound_penalty = max(0.0, float(np.quantile(tail_slice, 0.85) - (tail_mean + (0.045 - 0.008 * smallness))))
            bright_tail_penalty = max(0.0, tail_mean - (0.64 - 0.03 * size_alpha))

            local_grad = float(grad_prof[i])
            local_edge = float(edge_prof[i])
            local_skin = float(np.mean(skin_prof[max(0, i - 1):min(len(skin_prof), i + 2)]))

            inside_grad = float(np.mean(grad_prof[max(0, i - inside_w):i]))
            outside_grad = float(np.mean(tail_grad_slice[:max(2, outside_w)])) if tail_grad_slice.size >= 2 else 0.0
            outside_edge = float(np.mean(tail_edge_slice[:max(2, outside_w)])) if tail_edge_slice.size >= 2 else 0.0

            radius_frac = float(ray_rhos[i] / max(r, 1e-6))
            center_penalty = max(0.0, (0.44 + 0.10 * size_alpha) - radius_frac)
            rim_penalty = max(0.0, radius_frac - inner_stop_frac)
            relief_penalty = max(0.0, inside_grad - (outside_grad + (0.038 + 0.012 * smallness)))
            outer_bias = float(np.clip((radius_frac - (0.42 + 0.12 * size_alpha)) / (0.40 - 0.08 * size_alpha), 0.0, 1.0))
            weak_transition_penalty = (
                max(0.0, contrast_min - contrast)
                + max(0.0, local_drop_min - local_drop)
                + max(0.0, tail_dark_min - tail_dark)
                + max(0.0, persistent_dark_min - persistent_dark_frac)
            )

            hard_transition = (
                (contrast >= contrast_min)
                and (tail_dark >= tail_dark_min)
                and (persistent_dark_frac >= persistent_dark_min)
                and (local_drop >= local_drop_min or local_grad >= (0.066 + 0.010 * size_alpha) or local_edge >= (0.085 + 0.010 * size_alpha))
            )
            if not hard_transition:
                continue

            score = (
                0.20 * contrast
                + 0.13 * local_drop
                + 0.22 * tail_dark
                + 0.18 * persistent_dark_frac
                + 0.10 * persistent_tire_score
                + 0.05 * local_grad
                + 0.04 * local_edge
                + 0.05 * outside_dark
                + 0.04 * seed_match
                + 0.05 * outer_bias
                - 0.30 * local_skin
                - 0.10 * center_penalty
                - 0.09 * rim_penalty
                - 0.18 * relief_penalty
                - 0.20 * rebound_penalty
                - 0.15 * bright_tail_penalty
                - 0.22 * weak_transition_penalty
            )
            if score > best_score:
                best_score = score
                best_i = i

        if best_i is None or best_score < best_score_min:
            continue

        rho = float(ray_rhos[int(best_i)])
        px = float(cx + rho * math.cos(th))
        py = float(cy + rho * math.sin(th))
        rel_angle = float((th - peak_angle + math.pi) % (2.0 * math.pi) - math.pi)
        candidate_pts.append([px, py])
        candidate_w.append(float(best_score))
        candidate_angles.append(th)
        candidate_rhos.append(rho)
        candidate_rel.append(rel_angle)
        candidate_debug.append({"x": float(px / proc_scale), "y": float(py / proc_scale), "rho_frac": float(rho / max(r, 1e-6))})

    if len(candidate_pts) < 10:
        return {
            "found": False,
            "reason": "not enough radial boundary candidates",
            "candidate_count": int(len(candidate_pts)),
        }

    # Step 2b: neighbor-ray consensus. Keep points that are supported by nearby rays.
    pts_all = np.asarray(candidate_pts, dtype=np.float32)
    w_all = np.asarray(candidate_w, dtype=np.float32)
    rho_all = np.asarray(candidate_rhos, dtype=np.float32)
    rel_all = np.asarray(candidate_rel, dtype=np.float32)
    order = np.argsort(rel_all)
    pts_ord = pts_all[order]
    w_ord = w_all[order]
    rho_ord = rho_all[order]
    rel_ord = rel_all[order]

    win = int(max(2, min(5, 2 + round(1 + 2 * size_alpha))))
    sigma_rho = float(max(2.2, (0.060 - 0.010 * size_alpha) * r))
    neighbor_thresh = float(max(2.5, (0.080 - 0.010 * size_alpha) * r))
    min_neighbor_consistency = float(0.34 - 0.06 * smallness)

    neigh_cons = np.ones((len(rho_ord),), dtype=np.float32)
    for i in range(len(rho_ord)):
        lo = max(0, i - win)
        hi = min(len(rho_ord), i + win + 1)
        neigh = rho_ord[lo:hi]
        if neigh.size <= 2:
            continue
        local = np.delete(neigh, i - lo)
        if local.size == 0:
            continue
        local_med = float(np.median(local))
        dev = abs(float(rho_ord[i]) - local_med)
        smooth_score = math.exp(-dev / max(sigma_rho, 1e-6))
        density_score = float(np.mean(np.abs(local - rho_ord[i]) <= neighbor_thresh))
        neigh_cons[i] = float(np.clip(0.60 * smooth_score + 0.40 * density_score, 0.0, 1.0))

    keep = neigh_cons >= min_neighbor_consistency
    if int(np.count_nonzero(keep)) < 8:
        # Fallback: keep the most consistent rays if the filter is too harsh.
        topk = max(8, int(round(0.65 * len(rho_ord))))
        top_idx = np.argsort(neigh_cons)[-topk:]
        keep = np.zeros_like(neigh_cons, dtype=bool)
        keep[top_idx] = True

    pts = pts_ord[keep]
    w = w_ord[keep] * (0.55 + 0.45 * neigh_cons[keep])
    w = np.clip(w, 1e-4, None)
    kept_rel = rel_ord[keep]
    kept_rho = rho_ord[keep]
    neighbor_consistency_mean = float(np.mean(neigh_cons[keep])) if np.any(keep) else float(np.mean(neigh_cons))

    if len(pts) < 8:
        return {
            "found": False,
            "reason": "not enough consensus-supported boundary candidates",
            "candidate_count": int(len(candidate_pts)),
            "consensus_candidate_count": int(len(pts)),
        }

    # Step 3: robust chord fit with a mild expected-orientation prior.
    line_fit = robust_line_fit_ransac(
        pts,
        w,
        cx=float(cx),
        cy=float(cy),
        peak_angle=float(peak_angle),
        angle_tol_deg=max(22.0, float(cfg.chord_angle_refine_tol_deg) * 2.8),
        inlier_thresh=max(2.0, 0.030 * r),
        iterations=max(120, int(cfg.chord_ransac_iterations) * 2),
    )
    if line_fit is None:
        return {
            "found": False,
            "reason": "line fit failed",
            "candidate_count": int(len(candidate_pts)),
            "consensus_candidate_count": int(len(pts)),
        }

    nx = float(line_fit["normal"][0])
    ny = float(line_fit["normal"][1])
    line_d = float(line_fit["d"])
    inliers = np.asarray(line_fit["inliers"], dtype=bool)

    contact_vec = np.array([math.cos(peak_angle), math.sin(peak_angle)], dtype=np.float64)
    if nx * contact_vec[0] + ny * contact_vec[1] < 0.0:
        nx, ny = -nx, -ny
        line_d = -line_d

    alignment = float(max(0.0, nx * contact_vec[0] + ny * contact_vec[1]))
    if line_d <= 0.0:
        return {
            "found": False,
            "reason": "line landed on wrong side of center",
            "alignment": alignment,
        }
    if line_d >= r:
        return {
            "found": False,
            "reason": "line missed coin interior",
            "alignment": alignment,
        }

    geom = chord_endpoints_from_normal(cx, cy, r, nx, ny, line_d)
    if geom is None:
        return {"found": False, "reason": "failed to build chord"}
    chord_p0, chord_p1, chord_mid = geom

    depth_proc = float(r - line_d)
    depth_frac = float(depth_proc / max(r, 1e-6))
    if depth_proc < 1.0:
        return {"found": False, "reason": "chord estimate too shallow"}
    if depth_frac < float(cfg.chord_depth_min_frac) * 0.65 or depth_frac > min(0.96, float(cfg.chord_depth_max_frac) * 1.10):
        return {
            "found": False,
            "reason": "implausible chord depth",
            "depth_frac": depth_frac,
            "retake_recommended": True,
        }

    # Step 4: local side-band verification near the fitted chord.
    x0 = max(0, int(math.floor(cx - r - 3)))
    x1 = min(gray.shape[1], int(math.ceil(cx + r + 4)))
    y0 = max(0, int(math.floor(cy - r - 3)))
    y1 = min(gray.shape[0], int(math.ceil(cy + r + 4)))

    yy, xx = np.mgrid[y0:y1, x0:x1]
    dx = xx.astype(np.float32) - float(cx)
    dy = yy.astype(np.float32) - float(cy)
    rr = np.sqrt(dx * dx + dy * dy)
    signed = dx * float(nx) + dy * float(ny) - float(line_d)
    tx = float(-ny)
    ty = float(nx)
    tangent = dx * tx + dy * ty

    gray_roi = gray[y0:y1, x0:x1]
    grad_roi = grad[y0:y1, x0:x1]
    edge_roi = edges[y0:y1, x0:x1]
    skin_roi = skin[y0:y1, x0:x1]

    half_len = float(max(2.0, math.sqrt(max(0.0, r * r - line_d * line_d))))
    band_gap = float(max(1.4, (0.017 + 0.003 * size_alpha) * r))
    band_width = float(max(2.0, (0.052 + 0.008 * size_alpha) * r))
    along_margin = float(max(3.0, (0.08 + 0.02 * size_alpha) * r))
    inner_mask = rr <= float((0.93 + 0.01 * size_alpha) * r)
    along_mask = np.abs(tangent) <= float(min(0.96 * r, half_len + along_margin))
    local_mask = inner_mask & along_mask

    coin_side = local_mask & (signed <= -band_gap) & (signed >= -(band_gap + band_width))
    tire_side = local_mask & (signed >= band_gap) & (signed <= (band_gap + band_width))
    line_band = local_mask & (np.abs(signed) <= max(1.5, 0.018 * r))

    coin_count = int(np.count_nonzero(coin_side))
    tire_count = int(np.count_nonzero(tire_side))
    line_count = int(np.count_nonzero(line_band))

    support_frac = float(min(coin_count, tire_count) / max(12.0, 0.55 * half_len))
    support_frac = float(np.clip(support_frac, 0.0, 1.0))

    if coin_count >= 6 and tire_count >= 6:
        coin_mean = float(np.mean(gray_roi[coin_side]))
        tire_mean = float(np.mean(gray_roi[tire_side]))
        coin_grad = float(np.mean(grad_roi[coin_side]))
        tire_grad = float(np.mean(grad_roi[tire_side]))
        coin_edge = float(np.mean(edge_roi[coin_side]))
        tire_edge = float(np.mean(edge_roi[tire_side]))
        tire_skin = float(np.mean(skin_roi[tire_side]))
    else:
        coin_mean = tire_mean = 0.0
        coin_grad = tire_grad = 0.0
        coin_edge = tire_edge = 0.0
        tire_skin = 0.0

    if line_count >= 6:
        line_grad = float(np.mean(grad_roi[line_band]))
        line_edge = float(np.mean(edge_roi[line_band]))
    else:
        line_grad = 0.0
        line_edge = 0.0

    side_dark = float(np.clip(coin_mean - tire_mean, 0.0, 1.0))
    side_texture = float(np.clip((tire_grad - 0.60 * coin_grad) + 0.30 * (tire_edge - 0.50 * coin_edge), 0.0, 1.0))
    line_support = float(np.clip(0.65 * line_grad + 0.35 * line_edge, 0.0, 1.0))
    side_consistency = float(np.clip(
        0.46 * side_dark
        + 0.20 * side_texture
        + 0.18 * line_support
        + 0.12 * support_frac
        - 0.12 * tire_skin,
        0.0,
        1.0,
    ))

    inlier_count = int(np.count_nonzero(inliers))
    candidate_count = int(len(candidate_pts))
    consensus_candidate_count = int(len(pts))
    inlier_frac = float(inlier_count / max(consensus_candidate_count, 1))
    sector_coverage = float(point_sector_coverage(pts[inliers] if inlier_count >= 2 else pts, cx, cy, bins=24))
    mean_candidate_score = float(np.mean(w[inliers])) if inlier_count >= 1 else float(np.mean(w))

    confidence = float(np.clip(
        0.22 * inlier_frac
        + 0.16 * side_consistency
        + 0.14 * alignment
        + 0.11 * sector_coverage
        + 0.10 * mean_candidate_score
        + 0.09 * neighbor_consistency_mean
        + 0.08 * peak_val
        + 0.06 * line_support
        + 0.04 * support_frac,
        0.0,
        1.0,
    ))

    retake_recommended = bool(
        confidence < float(cfg.occlusion_retake_confidence)
        or inlier_count < 8
        or inlier_frac < 0.28
        or alignment < 0.34
        or neighbor_consistency_mean < 0.34
        or (side_consistency < 0.05 and line_support < 0.06)
    )
    if confidence < max(0.10, float(cfg.occlusion_hard_fail_confidence) * 0.80):
        return {
            "found": False,
            "reason": "low radial-scan confidence",
            "confidence": confidence,
            "retake_recommended": True,
        }

    scale_back = 1.0 / max(proc_scale, 1e-9)
    depth_px = float(depth_proc * scale_back)
    depth_mm = None
    if float(cfg.reference_diameter_mm) > 0.0:
        depth_mm = float(depth_proc * float(cfg.reference_diameter_mm) / max(2.0 * r, 1e-6))

    rim_pt = (float(cx + r * nx), float(cy + r * ny))
    meas_mid = ((rim_pt[0] + chord_mid[0]) * 0.5, (rim_pt[1] + chord_mid[1]) * 0.5)
    fill_area = circle_segment_area(r, depth_proc)

    return {
        "found": True,
        "score": float(confidence),
        "confidence": float(confidence),
        "retake_recommended": retake_recommended,
        "confidence_reason": "weak fit or weak side difference" if retake_recommended else "ok",
        "candidate_count": int(candidate_count),
        "consensus_candidate_count": int(consensus_candidate_count),
        "inlier_count": int(inlier_count),
        "inlier_fraction": float(inlier_frac),
        "point_sector_coverage": float(sector_coverage),
        "neighbor_consistency": float(neighbor_consistency_mean),
        "side_dark": float(side_dark),
        "side_texture": float(side_texture),
        "side_consistency": float(side_consistency),
        "line_support": float(line_support),
        "local_side_support": float(support_frac),
        "coin_side_count": int(coin_count),
        "tire_side_count": int(tire_count),
        "alignment": float(alignment),
        "contact_angle_deg": float(np.degrees(peak_angle) % 360.0),
        "depth_px": float(depth_px),
        "depth_mm": depth_mm,
        "depth_frac": float(depth_frac),
        "measurement_mode": "coin_inner_mask_radial_scan_ransac_chord_neighbor_consensus",
        "scale_adaptive_radius_px": float(r),
        "size_alpha": float(size_alpha),
        "measurement": {
            "p0": {"x": float(rim_pt[0] * scale_back), "y": float(rim_pt[1] * scale_back)},
            "p1": {"x": float(chord_mid[0] * scale_back), "y": float(chord_mid[1] * scale_back)},
            "mid": {"x": float(meas_mid[0] * scale_back), "y": float(meas_mid[1] * scale_back)},
            "angle_deg": float(np.degrees(math.atan2(ny, nx)) % 360.0),
        },
        "chord": {
            "p0": {"x": float(chord_p0[0] * scale_back), "y": float(chord_p0[1] * scale_back)},
            "p1": {"x": float(chord_p1[0] * scale_back), "y": float(chord_p1[1] * scale_back)},
            "mid": {"x": float(chord_mid[0] * scale_back), "y": float(chord_mid[1] * scale_back)},
            "angle_deg": float(np.degrees(math.atan2(chord_p1[1] - chord_p0[1], chord_p1[0] - chord_p0[0])) % 360.0),
        },
        "occluded_fill_area_px2": float(fill_area * (scale_back ** 2)),
        "boundary_points": candidate_debug,
    }


def mm_to_px_x(screen, mm: float) -> float:
    px_w = screen.size().width()
    mm_w = screen.physicalSize().width()
    if mm_w <= 0:
        dpi = screen.logicalDotsPerInchX()
        return (mm / 25.4) * dpi
    return (mm / mm_w) * px_w


class CircleCanvas(QWidget):
    def __init__(self, parent: Optional[QWidget] = None):
        super().__init__(parent)
        self.diameter_mm = 30.0
        self.margin = 30
        self.setMinimumSize(420, 420)

    def set_diameter_mm(self, mm: float) -> None:
        self.diameter_mm = float(mm)
        self.update()

    def paintEvent(self, _event) -> None:
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)

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

        painter.setPen(QPen(Qt.GlobalColor.black, 3))
        painter.setBrush(Qt.BrushStyle.NoBrush)
        painter.drawEllipse(rect)
        painter.drawLine(int(cx - r), int(cy), int(cx + r), int(cy))


class KeyringSizeDialog(QDialog):
    diameter_selected_mm = Signal(float)

    def __init__(self, initial_mm: float, parent: Optional[QWidget] = None):
        super().__init__(parent)
        self.setWindowTitle("Keyring Sizer")
        self.setModal(True)
        self.resize(720, 760)
        self.diameter_mm = float(initial_mm)

        outer = QVBoxLayout(self)
        outer.setContentsMargins(12, 12, 12, 12)
        outer.setSpacing(8)

        title = QLabel("Match the circle to your keyring's outer diameter.")
        title.setWordWrap(True)
        title.setStyleSheet("font-weight: 600;")
        outer.addWidget(title)

        self.canvas = CircleCanvas(self)
        outer.addWidget(self.canvas, stretch=1)

        self.readout = QLabel("")
        self.readout.setAlignment(Qt.AlignmentFlag.AlignCenter)
        outer.addWidget(self.readout)

        controls = QHBoxLayout()
        controls.setSpacing(10)
        outer.addLayout(controls)

        self.slider = QSlider(Qt.Orientation.Horizontal)
        self.slider.setToolTip("Adjust circle diameter (mm)")
        self.slider.setMinimum(100)
        self.slider.setMaximum(1016)
        self.slider.setValue(int(round(self.diameter_mm * 10)))
        self.slider.valueChanged.connect(self.apply_diameter_from_slider)
        controls.addWidget(self.slider, stretch=1)

        cancel_btn = QPushButton("Cancel")
        cancel_btn.clicked.connect(self.reject)
        controls.addWidget(cancel_btn)

        confirm_btn = QPushButton("Confirm")
        confirm_btn.clicked.connect(self.confirm_selection)
        controls.addWidget(confirm_btn)

        self.apply_diameter_from_slider()

    def apply_diameter_from_slider(self) -> None:
        mm = self.slider.value() / 10.0
        inches = mm / 25.4
        self.diameter_mm = mm
        self.canvas.set_diameter_mm(mm)
        self.readout.setText(f"Outer diameter: {inches:.2f} in ({mm:.2f} mm)")

    def confirm_selection(self) -> None:
        self.diameter_selected_mm.emit(float(self.diameter_mm))
        self.accept()


class ImageView(QLabel):
    def __init__(self):
        super().__init__()
        self.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.setMinimumSize(820, 620)
        self.setStyleSheet("background:#1f1f1f; border:1px solid #555;")
        self._img_bgr = None
        self._circle = None
        self._chord = None
        self._measure = None
        self._spotlight = None
        self._inner_circle = None

    def set_image(self, img_bgr: Optional[np.ndarray]):
        self._img_bgr = None if img_bgr is None else img_bgr.copy()
        self.update_view()

    def set_circle(self, circle: Optional[Tuple[float, float, float]]):
        self._circle = circle
        self.update_view()

    def set_occlusion(self, chord: Optional[Tuple[Tuple[float, float], Tuple[float, float]]], measure: Optional[Tuple[Tuple[float, float], Tuple[float, float]]], inner_circle: Optional[Tuple[float, float, float]] = None):
        self._chord = chord
        self._measure = measure
        self._inner_circle = inner_circle
        self.update_view()

    def set_spotlight(self, spotlight):
        self._spotlight = spotlight
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
            painter.drawEllipse(int(round(cx - r)), int(round(cy - r)), int(round(2 * r)), int(round(2 * r)))

        if self._inner_circle is not None:
            icx, icy, ir = self._inner_circle
            painter.setPen(QPen(QColor(210, 80, 255), 10))
            painter.drawEllipse(int(round(icx - ir)), int(round(icy - ir)), int(round(2 * ir)), int(round(2 * ir)))

        if self._spotlight is not None and self._circle is not None:
            ang_deg, half_deg = self._spotlight
            cx, cy, r = self._circle
            a0 = math.radians(ang_deg - half_deg)
            a1 = math.radians(ang_deg + half_deg)
            p0 = (cx + r * math.cos(a0), cy + r * math.sin(a0))
            p1 = (cx + r * math.cos(a1), cy + r * math.sin(a1))
            rect_x = int(round(cx - r))
            rect_y = int(round(cy - r))
            rect_w = int(round(2 * r))
            rect_h = int(round(2 * r))
            start16 = int(round(-(ang_deg + half_deg) * 16))
            span16 = int(round(2 * half_deg * 16))

            painter.setPen(Qt.PenStyle.NoPen)
            painter.setBrush(QColor(40, 220, 120, 70))
            painter.drawPie(rect_x, rect_y, rect_w, rect_h, start16, span16)

            painter.setBrush(Qt.BrushStyle.NoBrush)
            painter.setPen(QPen(QColor(40, 220, 120), 6))
            painter.drawLine(int(round(cx)), int(round(cy)), int(round(p0[0])), int(round(p0[1])))
            painter.drawLine(int(round(cx)), int(round(cy)), int(round(p1[0])), int(round(p1[1])))
            painter.drawArc(rect_x, rect_y, rect_w, rect_h, start16, span16)

        if self._chord is not None:
            p0, p1 = self._chord
            painter.setPen(QPen(QColor(255, 145, 0), 10))
            painter.drawLine(int(round(p0[0])), int(round(p0[1])), int(round(p1[0])), int(round(p1[1])))

        if self._measure is not None:
            p0, p1 = self._measure
            painter.setPen(QPen(QColor(255, 95, 0), 8))
            painter.drawLine(int(round(p0[0])), int(round(p0[1])), int(round(p1[0])), int(round(p1[1])))

        painter.end()
        pix = QPixmap.fromImage(canvas)
        self.setPixmap(pix.scaled(self.size(), Qt.AspectRatioMode.KeepAspectRatio, Qt.TransformationMode.SmoothTransformation))

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
        self.setWindowTitle("Tire Tread Detector v3")
        self.slider_scale = 10.0
        self.img_bgr: Optional[np.ndarray] = None
        self.cfg = CircleSearchConfig()
        self.last_occlusion = None
        self.locked_circle = None
        self.keyring_diameter_mm = 38.10

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

        controls_layout.addWidget(QLabel("Reference"))

        ref_row = QHBoxLayout()
        self.coin_radio = QRadioButton("Coin")
        self.keyring_radio = QRadioButton("Keyring")
        self.coin_radio.setChecked(True)
        ref_row.addWidget(self.coin_radio)
        ref_row.addWidget(self.keyring_radio)
        ref_row.addStretch(1)
        controls_layout.addLayout(ref_row)

        self.reference_group = QButtonGroup(self)
        self.reference_group.addButton(self.coin_radio)
        self.reference_group.addButton(self.keyring_radio)
        self.coin_radio.toggled.connect(self.on_reference_mode_change)
        self.keyring_radio.toggled.connect(self.on_reference_mode_change)

        self.coin_box = QComboBox()
        self.coin_box.addItems(["Quarter", "Nickel", "Penny", "Dime"])
        self.coin_box.currentTextChanged.connect(self.coin_changed)
        controls_layout.addWidget(self.coin_box)

        self.keyring_size_btn = QPushButton("Size Keyring")
        self.keyring_size_btn.clicked.connect(self.open_keyring_sizer)
        controls_layout.addWidget(self.keyring_size_btn)

        self.reference_info_label = QLabel("")
        self.reference_info_label.setWordWrap(True)
        controls_layout.addWidget(self.reference_info_label)

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

        detect_btn = QPushButton("Detect Object")
        detect_btn.clicked.connect(self.run_detection)
        controls_layout.addWidget(detect_btn)

        controls_layout.addWidget(QLabel("Occlusion direction"))
        self.direction_slider = self._make_slider(0, 3600, 900)
        self.direction_slider.setEnabled(False)
        self.direction_label = QLabel("90.0°")
        self.direction_slider.valueChanged.connect(self.direction_changed)
        controls_layout.addWidget(SliderRow("Angle", self.direction_slider, self.direction_label))

        self.occlusion_btn = QPushButton("Submit direction / run occlusion")
        self.occlusion_btn.setEnabled(False)
        self.occlusion_btn.clicked.connect(self.run_occlusion_from_direction)
        controls_layout.addWidget(self.occlusion_btn)

        self.result_label = QLabel("Depth: —")
        self.result_label.setWordWrap(True)
        controls_layout.addWidget(self.result_label)

        controls_layout.addStretch(1)
        self.resize(1420, 860)
        self._update_reference_controls(clear_detection=False)
        self.direction_changed()

    def _make_slider(self, lo, hi, value):
        s = QSlider(Qt.Orientation.Horizontal)
        s.setRange(lo, hi)
        s.setValue(value)
        s.setSingleStep(1)
        s.setPageStep(10)
        s.setTickInterval(1)
        return s

    def coin_changed(self, _name: str):
        self._update_reference_controls(clear_detection=True)

    def on_reference_mode_change(self):
        self._update_reference_controls(clear_detection=True)

    def open_keyring_sizer(self):
        if not self.keyring_radio.isChecked():
            return
        dialog = KeyringSizeDialog(self.keyring_diameter_mm, self)
        dialog.diameter_selected_mm.connect(self.set_keyring_diameter_mm)
        dialog.exec()

    def set_keyring_diameter_mm(self, mm: float) -> None:
        self.keyring_diameter_mm = float(mm)
        self._update_reference_controls(clear_detection=True)

    def _update_reference_controls(self, clear_detection: bool = False):
        mode = "keyring" if self.keyring_radio.isChecked() else "coin"
        self.cfg.reference_mode = mode

        self.coin_box.setEnabled(mode == "coin")
        self.coin_box.setVisible(mode == "coin")
        self.keyring_size_btn.setEnabled(mode == "keyring")
        self.keyring_size_btn.setVisible(mode == "keyring")

        if mode == "coin":
            coin_name = self.coin_box.currentText().strip() or "Quarter"
            if coin_name not in COIN_DIAMETERS_MM:
                coin_name = "Quarter"
                self.coin_box.setCurrentText(coin_name)
            self.cfg.reference_diameter_mm = float(COIN_DIAMETERS_MM[coin_name])
            self.cfg.forced_sector_half_deg = MANUAL_SECTOR_HALF_DEG.get(coin_name, 52.0)
            self.reference_info_label.setText(
                f"Coin selected: {coin_name}\nReference diameter: {self.cfg.reference_diameter_mm:.2f} mm"
            )
        else:
            self.cfg.reference_diameter_mm = float(self.keyring_diameter_mm)
            self.reference_info_label.setText(
                f"Keyring outer diameter: {self.keyring_diameter_mm / 25.4:.2f} in ({self.keyring_diameter_mm:.2f} mm)"
            )

        if clear_detection:
            self.last_occlusion = None
            self.locked_circle = None
            self.view.set_occlusion(None, None)
            self.direction_slider.setEnabled(False)
            self.occlusion_btn.setEnabled(False)
            self.result_label.setText("Depth: Detect the reference, then set the occlusion direction.")

        self.overlay_changed()

    @staticmethod
    def slider_deg_to_image_deg(slider_deg: float) -> float:
        return (-float(slider_deg)) % 360.0

    def current_circle(self):
        return (
            float(self.slider_x.value()) / self.slider_scale,
            float(self.slider_y.value()) / self.slider_scale,
            float(self.slider_r.value()) / self.slider_scale,
        )

    def load_image(self):
        path, _ = QFileDialog.getOpenFileName(self, "Select image", "", "Images (*.png *.jpg *.jpeg *.bmp *.webp)")
        if not path:
            return
        img = cv2.imread(path)
        if img is None:
            QMessageBox.warning(self, "Load failed", "Could not read that image.")
            return

        self.img_bgr = img
        self.last_occlusion = None
        self.locked_circle = None
        h, w = img.shape[:2]
        self.slider_x.setRange(0, int((w - 1) * self.slider_scale))
        self.slider_y.setRange(0, int((h - 1) * self.slider_scale))
        self.slider_r.setRange(int(5 * self.slider_scale), int(max(10, min(w, h) // 2) * self.slider_scale))
        self.slider_x.setValue(int((w / 2) * self.slider_scale))
        self.slider_y.setValue(int((h / 2) * self.slider_scale))
        self.slider_r.setValue(int(max(10, min(w, h) // 8) * self.slider_scale))
        self.view.set_image(img)
        self.view.set_occlusion(None, None)
        self.view.set_spotlight(None)
        self.direction_slider.setEnabled(False)
        self.occlusion_btn.setEnabled(False)
        self.result_label.setText("Depth: Detect the reference, then set the occlusion direction.")
        self.overlay_changed()

    def overlay_changed(self):
        if self.img_bgr is None:
            return
        cx, cy, r = self.current_circle()
        self.label_x.setText(f"{cx:.1f}")
        self.label_y.setText(f"{cy:.1f}")
        self.label_r.setText(f"{r:.1f}")
        self.view.set_circle((cx, cy, r))
        if self.direction_slider.isEnabled():
            self.view.set_spotlight((self.slider_deg_to_image_deg(self.direction_slider.value() / 10.0), self.cfg.forced_sector_half_deg))

    def direction_changed(self):
        ang = self.direction_slider.value() / 10.0
        self.direction_label.setText(f"{ang:.1f}°")
        if self.direction_slider.isEnabled():
            self.view.set_spotlight((self.slider_deg_to_image_deg(ang), self.cfg.forced_sector_half_deg))

    def run_detection(self):
        if self.img_bgr is None:
            return

        cx, cy, r = self.current_circle()
        circle = detect_circle_fast(self.img_bgr, cx, cy, r)
        if circle is None:
            QMessageBox.information(self, "No circle found", "No confident circle found. Move the sliders closer to the object and try again.")
            return

        x, y, rr, _score = circle
        self.locked_circle = {"cx": x, "cy": y, "r": rr}
        self.slider_x.setValue(int(round(x * self.slider_scale)))
        self.slider_y.setValue(int(round(y * self.slider_scale)))
        self.slider_r.setValue(int(round(rr * self.slider_scale)))
        self.view.set_occlusion(None, None)
        self.direction_slider.setEnabled(True)
        self.occlusion_btn.setEnabled(True)
        self.direction_changed()
        self.result_label.setText("Depth: Reference locked. Rotate the spotlight toward the tire occlusion, then click Submit direction / run occlusion.")


    def run_occlusion_from_direction(self):
        if self.img_bgr is None or self.locked_circle is None:
            return
        self.cfg.forced_occlusion_angle_deg = self.slider_deg_to_image_deg(self.direction_slider.value() / 10.0)
        context = make_occlusion_context(self.img_bgr)
        occ = detect_occlusion_from_circle(context, self.locked_circle, self.cfg)
        self.last_occlusion = occ

        chord = None
        measure = None
        if occ.get("found") and occ.get("chord") and occ.get("measurement"):
            cp0 = occ["chord"]["p0"]["x"], occ["chord"]["p0"]["y"]
            cp1 = occ["chord"]["p1"]["x"], occ["chord"]["p1"]["y"]
            mp0 = occ["measurement"]["p0"]["x"], occ["measurement"]["p0"]["y"]
            mp1 = occ["measurement"]["p1"]["x"], occ["measurement"]["p1"]["y"]
            chord = (cp0, cp1)
            measure = (mp0, mp1)

        inner_circle = None
        if occ.get("inner_circle"):
            inner = occ["inner_circle"]
            inner_circle = (inner["cx"], inner["cy"], inner["r"])
        self.view.set_occlusion(chord, measure, inner_circle)

        if occ.get("found"):
            mm = occ.get("depth_mm")
            if isinstance(mm, (int, float)):
                mm_text = f"{mm:.2f} mm"
                depth_32 = (float(mm) / 25.4) * 32.0
                depth_32_text = f"{depth_32:.2f}/32\""
                self.result_label.setText(f"Depth: {mm_text} | {depth_32_text}")
            else:
                self.result_label.setText("Depth: —")
        else:
            self.result_label.setText("Depth: —")



def main():
    app = QApplication(sys.argv)
    win = MainWindow()
    win.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
