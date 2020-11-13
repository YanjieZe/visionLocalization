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
    gray = np.float32(gray)
    blockSize=5
    ksize=3
    k=0.04
    dst = cv2.cornerHarris(gray,blockSize,ksize,k)
    return dst