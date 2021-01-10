import cv2
import mvsdk
import mvcamera

def getCropped(picture, label):
    
    # label格式：class ，x_center ，y_center ，width， height
    # 对应：      0       1            2        3       4
    width = picture.shape[0]
    height = picture.shape[1]
    # 计算要裁剪的四角
    x0 = width*(label[1] - 0.5*label[3])
    x1 = int(width*(label[1] - 0.5*label[3]))
    print("误差：", x0 - x1)# 误差
    x2 = int(width*(label[1] + 0.5*label[3]))
    y1 = int(height*(label[2] - 0.5*label[4]))
    y2 = int(height*(label[2] + 0.5*label[4]))
    '''
    因为是像素化的图片，用int做了一个近似，存在一定误差
    '''
    img_cropped = picture[x1:x2,y1:y2]

    return img_cropped


def findCorner(img):
    img_blur = cv2.GaussianBlur(img, (3,3),0)
    threshold1 = 100
    threshold2 = 250
    img_canny = cv2.Canny(img_blur,threshold1,threshold2)
    # retr_tree的方式检索轮廓
    _, contours, hierarchy = cv2.findContours(img_canny, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)


    '''
    备注一下：
    hierarchy的内容： [Next, Previous, First_Child, Parent]
    '''

    '''
    找到面积最大的三个轮廓
    '''
    area1 = 0
    area2 = 0
    area3 = 0
    idx1 = -1
    idx2 = -1
    idx3 = -1
    if len(contours)<3:
        '''
        这里等下要重写，改成能重新调整阈值进行匹配
        '''
        raise "contours not enough"

    for idx,cnt in enumerate(contours):
        area = cv2.contourArea(cnt)
        if area > area1:
            area3 = area2
            idx3 = idx2
            area2 = area1
            idx2 = idx1
            area1 = area
            idx1 = idx
        elif area > area2:
            area3 = area2
            idx3 = idx2
            area2 = area
            idx2 = idx
        elif area > area3:
            area3 = area
            idx3 = idx
       
    for i in [idx1,idx2,idx3]:
        #计算轮廓中心点
        M=cv2.moments(contours[i])
        
        cx = int(M['m10']/M['m00'])
        cy = int(M['m01']/M['m00'])
        img = cv2.circle(img,(cx,cy),1,[255,0,0],5)
        center_point = (cx,cy)
        img = cv2.putText(img,"(%d,%d)"%(cx,cy),center_point,cv2.FONT_HERSHEY_PLAIN,0.5,(0,0,255))
        img = cv2.drawContours(img, contours,i,[255,255,0],2)
    # 这里是找到了中心字母和字符的轮廓了，并且标了出来


    '''
    通过几何形状逼近找出右下角小方块
    '''
    for i in range(len(contours)):
        # 轮廓逼近
        epsilon = 0.01 * cv2.arcLength(contours[i], True)
        approx = cv2.approxPolyDP(contours[i], epsilon, True)
        # 分析几何形状
        corners = len(approx)
        shape_type = ""
        if corners == 4:
            M=cv2.moments(contours[i])
            cx = int(M['m10']/M['m00'])
            cy = int(M['m01']/M['m00'])
            img = cv2.circle(img,(cx,cy),1,[255,0,0],5)
            center_point = (cx,cy)
            img = cv2.putText(img,"(%d,%d)"%(cx,cy),center_point,cv2.FONT_HERSHEY_PLAIN,0.5,(0,0,255))
            img = cv2.drawContours(img, contours,i,[255,255,0],2)
            



    return img


def cameraLoop():
        DevList = mvsdk.CameraEnumerateDevice()
        nDev = len(DevList)
        if nDev < 1:
            print("No camera was found!")
            return

        for i, DevInfo in enumerate(DevList):
            print("{}: {} {}".format(i, DevInfo.GetFriendlyName(), DevInfo.GetPortType()))

        cams = []
        for i in map(lambda x: int(x), raw_input("Select cameras: ").split()):
            cam = Camera(DevList[i])
            if cam.open():
                cams.append(cam)

        while (cv2.waitKey(1) & 0xFF) != ord('q'):
            for cam in cams:
                frame = cam.grab()
                if frame is not None:
                    frame = cv2.resize(frame, (640,480), interpolation = cv2.INTER_LINEAR)
                    img = findCorner(frame)
                    cv2.imshow("{} Press q to end".format(cam.DevInfo.GetFriendlyName()), frame)

        for cam in cams:
            cam.close()


if __name__=="__main__":
    mode = "camera"

    if mode == "img":
        '''
        读取label
        '''
        labelfile = open('36.txt','r',encoding='utf-8')
        # label格式：class ，x_center ，y_center ，width， height
        # 对应：      0       1            2        3       4
        label = labelfile.readline().split(' ')
        labelfile.close()
        float_label = []
        for num in  label:
            float_label.append(float(num))
        label = float_label

        '''
        读取picture
        '''
        picture = cv2.imread('36.jpg')

        img = getCropped(picture, label)
        img = findCorner(img)
        '''
        debug区
        '''
        cv2.imshow('croped', img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    elif mode=="camera":
        try:
            cameraLoop()
        finally:
            cv2.destroyAllWindows()
