
import cv2
import cv2.aruco as aruco
import numpy as np
import os
import depthai as dai
def findArucoMarkers(img, markerSize =4, totalMarkers=50, draw=True):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    key = getattr(aruco, f'DICT_{markerSize}X{markerSize}_{totalMarkers}')
    arucoDict = aruco.Dictionary_get(key)
    arucoParam = aruco.DetectorParameters_create()
    bboxs, ids, rejected = aruco.detectMarkers(gray, arucoDict, parameters = arucoParam)
    return (bboxs)

img0 = cv2.imread('picture1.jpg')
img1 = cv2.imread('picture2.jpg')

#disparity equation d = basline(8cm) * focallength(1.636331765375964e+03)/(ul - ur)

baseline = 11.5
focal_length = 1.636331765375964e+03

#we use the right corners of aruco markers
bbox1= findArucoMarkers(img0)
bbox2 = findArucoMarkers(img1)

print(bbox1[0][0][3][0])
print(bbox2[0][0][3][0])

d = (baseline * focal_length)/(bbox1[0][0][3][0]-bbox2[0][0][3][0])
print(d)
# disparity = 155 cm
# actual measurement = 153 cm