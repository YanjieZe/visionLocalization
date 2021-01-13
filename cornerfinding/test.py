import numpy as np
import cv2

img = cv2.imread("example2.jpg",0)
img_origin = img.copy()
img = cv2.GaussianBlur(img,(3,3),0)
img = cv2.adaptiveThreshold(img,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY,21,10)

_,contours,hie = cv2.findContours(img,cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
im = cv2.drawContours(img_origin,contours,-1,[0,255,0],2)
cv2.imshow("img",im)
cv2.waitKey(0)