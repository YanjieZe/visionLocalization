import cv2
import os

def getPicture(num,epoch):
    cap = cv2.VideoCapture(0)
    assert cap.isOpened(), 'Cannot capture source'
    for i in range(epoch):
        for j in range(num):
            ret,frame = cap.read()
            cv2.imwrite('rmset/'+str(epoch)+'/'+str(num)+'.jpg',frame)
            key = cv2.waitKey(delay=10)
            if key == ord('q'):
                break
        if key == ord('q'):
                break
            
    
if __name__="__main__":
    num=1
    epoch=1
    getPicture(num,epoch)