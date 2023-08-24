import cv2
import numpy as np
#import imutils
import math
import copy

def CannyEdge(capturedImage):  
    grayScale = cv2.cvtColor(capturedImage, cv2.COLOR_BGR2GRAY)
    bilateralFilter = cv2.bilateralFilter(grayScale, 11, 17, 17)
    imageMedian = np.median(capturedImage)
    lowerThreshold = max(0, (0.7 * imageMedian))
    upperThreshold = min(255, (0.7 * imageMedian)) 
    cannyEdgeImage = cv2.Canny(bilateralFilter,lowerThreshold,upperThreshold)
    return cannyEdgeImage

def BlackWhite(capturedImage):
    hsv = cv2.cvtColor(capturedImage, cv2.COLOR_BGR2HSV)
    lowerWhite = np.array([0,0,168])
    upperWhite = np.array([172,111,255])
    mask = cv2.inRange(hsv, lowerWhite, upperWhite)
    masked = cv2.bitwise_and(capturedImage,capturedImage, mask= mask)
    (thresh, blackWhiteImage) = cv2.threshold(masked, 127, 255, cv2.THRESH_BINARY)
    return blackWhiteImage

def LaneDetection_(cannyEdgeImage, capturedImage):
    (cnts, _) = cv2.findContours(cannyEdgeImage, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    cnts=sorted(cnts, key = cv2.contourArea, reverse = True)[:]
    for c in cnts:
        perimeter = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, 0.02 * perimeter, True) 
        if (len(approx) == 4):  # Select the contour with 4 corners
            Number_Plate_Area = approx
            x,y,w,h = cv2.boundingRect(Number_Plate_Area)
            # start_point = (y,x)
            # end_point = (y+h,x+w)
            color = (0, 255, 0)
            thickness = 9
            start_point = (x,y)
            end_point = (x+w, y+h)
            image = cv2.rectangle(capturedImage, start_point, end_point, color, thickness)

    return image


def LaneDetection(cannyEdgeImage, capturedImage):
    linesList =[]
    lines = cv2.HoughLinesP(
                cannyEdgeImage, # Input edge image
                1, # Distance resolution in pixels
                np.pi/180, # Angle resolution in radians
                threshold=150, # Min number of votes for valid line
                minLineLength=1, # Min allowed length of line
                maxLineGap=50 # Max allowed gap between line for joining them
                )
    
    # Iterate over points
    for points in lines:
        x1,y1,x2,y2=points[0]
        image  = cv2.line(capturedImage,(x1,y1),(x2,y2),(0,255,0),9)
        cv2.imshow("lines", image)

        linesList.append([(x1,y1),(x2,y2)])
        # cv2.waitKey(0)
    
    return image, linesList
    

# capturedImage = cv2.imread('1.jpg')
# # capturedImage = cv2.imread('Crossraod.jpg')
# # capturedImage = cv2.imread('test.jpg')
# image = copy.deepcopy(capturedImage)
# blackWhiteImage = BlackWhite(image)
# cv2.imshow('blackWhiteImage',blackWhiteImage)
# cannyEdgeImage = CannyEdge(blackWhiteImage)
# cv2.imshow("cannyEdgeImage", cannyEdgeImage)
# image = copy.deepcopy(capturedImage)
# lanes, lines = LaneDetection(cannyEdgeImage, image)
# for [p1,p2] in lines:
#     if(abs(p1[0]-p2[0]) > abs(p1[1]-p2[1])):
#         print(p1,p2," horizontal line")
#     elif(abs(p1[0]-p2[0]) < abs(p1[1]-p2[1])):
#         print(p1,p2," vertical line")
# cv2.imshow("lanes", lanes)
# cv2.imwrite("output.jpg",lanes)
# cv2.waitKey(0)

for i in range(1,5):
    capturedImage = cv2.imread(str(i)+'.jpg')
    # capturedImage = cv2.imread('Crossraod.jpg')
    # capturedImage = cv2.imread('test.jpg')
    image = copy.deepcopy(capturedImage)
    blackWhiteImage = BlackWhite(image)
    cv2.imshow('blackWhiteImage',blackWhiteImage)
    cannyEdgeImage = CannyEdge(blackWhiteImage)
    cv2.imshow("cannyEdgeImage", cannyEdgeImage)
    image = copy.deepcopy(capturedImage)
    lanes, lines = LaneDetection(cannyEdgeImage, image)
    for [p1,p2] in lines:
        if(abs(p1[0]-p2[0]) > abs(p1[1]-p2[1])):
            print(p1,p2," horizontal line")
        elif(abs(p1[0]-p2[0]) < abs(p1[1]-p2[1])):
            print(p1,p2," vertical line")
    cv2.imshow("lanes", lanes)
    cv2.imwrite("output"+str(i)+".jpg",lanes)

    cv2.waitKey(0)
