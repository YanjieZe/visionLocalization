import cv2

path = 0
cap = cv2.VideoCapture(path)
count = 0
while cap.isOpened():
    ret, frame = cap.read()
    print('frame.shape:', frame.shape)
    cv2.imshow('frame', frame)

    
    key = cv2.waitKey(delay=20)
    if key == ord('s'):
        cv2.imwrite("rmset/"+str(count)+ '.jpg', frame)
        count+=1
    
    if key == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()