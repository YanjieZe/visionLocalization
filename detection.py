import argparse
import os
import platform
import sys
import time

import cv2
import function
import numpy as np
parser = argparse.ArgumentParser(description='corner point detection')

parser.add_argument('--mode',dest='mode', type=str,required=True,
                    help='choose the mode, webCam or figure or video')
parser.add_argument('--path', dest='path', type=str, required=False, 
                    help='the path of inputfile, figure or video')
args = parser.parse_args()


if __name__ == "__main__":
    if args.mode == "figure":

        img = cv2.imread(args.path)
     
        dst = function.HarrisCornerDetection(img)

        function.figshow("origin",img)
        function.figshow("dst", dst)
        
    if args.mode == "webcam":
        cap = cv2.VideoCapture(0)
        assert cap.isOpened(), 'Cannot capture source'
        while (True):
            ret, frame = cap.read()
            dst = function.HarrisCornerDetection(frame)
            cv2.imshow('camera',dst)
            if cv2.waitKey(1) & 0xFF == ord('q'): 
                break
        cap.release()
        cv2.destroyAllWindows()