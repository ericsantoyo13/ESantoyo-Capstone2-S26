import matplotlib.pyplot as plt
import cv2 as cv
import numpy as np
import os

def houghCircleTransform():
    root = os.getcwd()
    #imgPath = os.path.join(root, 'keyring.jpeg')
    #imgPath = os.path.join(root, 'quarter.jpeg')
    imgPath = os.path.join(root, 'dime.jpeg')

    img = cv.imread(imgPath)
    imgRGB = cv.cvtColor(img, cv.COLOR_BGR2RGB)
    imgGray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    imgGray = cv.medianBlur(imgGray,21)
    
    # Tweaking parameters:
    # param1: Threshold for edge detectors. Increase --> Edge detection less sensitive and reduce noise
    # minDist: Set minimum distance between centerpoints of circles
    # dp: Resolution of search grid for detecting circles. dp = 1 more precise. dp = 2 faster and less sensitive to noise.
    # param2: Threshold for what gets considered a circle. Increase --> Fewer circles detected
    param1 = 300
    minDist = 100
    dp = 1
    param2 = 15
    minRadius = 250
    maxRadius = 400

    circles = cv.HoughCircles(imgGray, cv.HOUGH_GRADIENT, dp=dp, minDist=minDist, param1=param1,
    param2=param2, minRadius=minRadius, maxRadius=maxRadius)
    circles = np.uint16(np.around(circles))

    # Find center of image
    height, width = img.shape[:2]
    center_x = width // 2
    center_y = height // 2

    # Keep any hough circles that have a center this close to the center of the image
    keep_circles_inside_radius = 70
    cv.circle(imgRGB, (center_x, center_y),keep_circles_inside_radius,(200,0,0),3)

    # Draw upper and lower bounds for possible circles
    cv.circle(imgRGB, (center_x, center_y),minRadius,(200,200,0),10)
    cv.circle(imgRGB, (center_x, center_y),maxRadius,(200,200,0),10)

    # Draw circles that meet criteria
    for circle in circles[0,:]:
        if abs(center_x - circle[0]) < keep_circles_inside_radius and abs(center_y - circle[1]) < keep_circles_inside_radius:
            print(circle[0],circle[1],circle[2])
            cv.circle(imgRGB,(circle[0],circle[1]),circle[2],(0,0,255),10) # Circle
            cv.circle(imgRGB,(circle[0],circle[1]),1,(0,0,255),15) # Centerpoint
        #cv.circle(imgRGB,(circle[0],circle[1]),circle[2],(0,0,255),10)



    plt.figure()
    plt.imshow(imgRGB)
    #plt.imshow(imgGray)
    plt.show()

if __name__ == "__main__":
    houghCircleTransform()
    

