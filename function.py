import cv2
import numpy as np

def figshow(imgname,img):
    if not isinstance(imgname,str):
        raise Exception("Img name must be string type")

    cv2.imshow(imgname,img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def HarrisCornerDetection(Img):
    gray = cv2.cvtColor(Img,cv2.COLOR_BGR2GRAY)

    thresh = 200 # hyperparameter
    _, threshImg = cv2.threshold(gray,thresh,255,0)
    dst, contours, _ = cv2.findContours(threshImg,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)

    dst = cv2.drawContours(dst,contours, -1, (0,0,255), 3)
    # dst = np.float32(dst)
    # blockSize=15
    # ksize=5
    # k=0.04
    # dst = cv2.cornerHarris(dst,blockSize,ksize,k)

    return dst

def countours(Img):
    gray = cv2.cvtColor(Img,cv2.COLOR_BGR2GRAY)
    dst = cv2.findContours(gray)
    dst = cv2.drawContours(dst,color='red')

    return dst