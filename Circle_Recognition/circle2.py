import matplotlib.pyplot as plt
import cv2 as cv
import numpy as np
import os

def circle_edge_support_score(edges, cx, cy, r, band=3, n_samples=360):
    h, w = edges.shape
    angles = np.linspace(0, 2*np.pi, n_samples, endpoint=False)

    xs = (cx + r * np.cos(angles)).astype(np.int32)
    ys = (cy + r * np.sin(angles)).astype(np.int32)

    inb = (xs >= 0) & (xs < w) & (ys >= 0) & (ys < h)
    xs = xs[inb]
    ys = ys[inb]
    if len(xs) == 0:
        return 0.0

    hits = 0
    for x, y in zip(xs, ys):
        x0 = max(0, x - band); x1 = min(w, x + band + 1)
        y0 = max(0, y - band); y1 = min(h, y + band + 1)
        if np.any(edges[y0:y1, x0:x1]):
            hits += 1

    return hits / len(xs)

def houghCircleTransform():
    root = os.getcwd()
    #imgPath = os.path.join(root, 'keyring.jpeg')
    #imgPath = os.path.join(root, 'quarter.jpeg')
    #imgPath = os.path.join(root, 'dime.jpeg')
    #imgPath = os.path.join(root, "quarter.jpeg")
    imgPath = os.path.join(root, "dime_off.jpeg")


    img = cv.imread(imgPath)

    imgRGB = cv.cvtColor(img, cv.COLOR_BGR2RGB)

    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    gray_blur = cv.medianBlur(gray, 21)

    edges = cv.Canny(gray_blur, 60, 180)

    # Tweaking parameters:
    # param1: Threshold for edge detectors. Increase --> Edge detection less sensitive and reduce noise
    # minDist: Set minimum distance between centerpoints of circles
    # dp: Resolution of search grid for detecting circles. dp = 1 more precise. dp = 2 faster and less sensitive to noise.
    # param2: Threshold for what gets considered a circle. Increase --> Fewer circles detected

    param1 = 300
    minDist = 5
    dp = 1
    param2 = 15
    minRadius = 100
    maxRadius = 500

    circles = cv.HoughCircles(
        gray_blur, cv.HOUGH_GRADIENT,
        dp=dp, minDist=minDist,
        param1=param1, param2=param2,
        minRadius=minRadius, maxRadius=maxRadius
    )

    if circles is None:
        print("No circles found.")
        plt.figure()
        plt.title("No circles found")
        plt.imshow(imgRGB)
        plt.show()
        return

    circles = np.uint16(np.around(circles))[0, :]

# Score circles based on how many hits they get
    scored = []
    h, w = edges.shape
    border_margin = 5

    for (cx, cy, r) in circles:
        cx, cy, r = int(cx), int(cy), int(r)

        # Skip circles that are close to the image borders
        if cx - r < border_margin or cy - r < border_margin or cx + r >= w - border_margin or cy + r >= h - border_margin:
            continue

        score = circle_edge_support_score(edges, cx, cy, r, band=3, n_samples=400)
        scored.append((cx, cy, r, score))

    if not scored:
        print("All circles are too close to the border.")
        plt.figure()
        plt.imshow(imgRGB)
        plt.show()
        return

    best = max(scored, key=lambda t: t[3])
    cx_best, cy_best, r_best, s_best = best
    print(f"Best (by edge support): cx={cx_best}, cy={cy_best}, r={r_best}, support={s_best:.3f}")

    # Among circles with almost the same center, choose to show the one with the largest radius
    center_tol = 12  
    min_support = 0.30 

    same_object = []
    for (cx, cy, r, s) in scored:
        if abs(cx - cx_best) <= center_tol and abs(cy - cy_best) <= center_tol and s >= min_support:
            same_object.append((cx, cy, r, s))

    if same_object:
        chosen = max(same_object, key=lambda t: t[2])
    else:
        chosen = best

    cx, cy, r, s = chosen
    print(f"Chosen: cx={cx}, cy={cy}, r={r}, support={s:.3f}, candidates_in_cluster={len(same_object)}")


    cv.circle(imgRGB, (cx, cy), r, (0, 0, 255), 10)
    cv.circle(imgRGB, (cx, cy), 2, (0, 0, 255), 15)

    plt.figure()
    plt.title("Detected circle")
    plt.imshow(imgRGB)

    plt.figure()
    plt.title("Canny edges")
    plt.imshow(edges, cmap="gray")

    plt.show()


if __name__ == "__main__":
    houghCircleTransform()