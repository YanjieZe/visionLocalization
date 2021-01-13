import cv2
import mvsdk
import mvcamera
import numpy as np
import argparse
import time
from apriltag import Apriltag

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



'''
根据中心字符位置进行判断的角点检测算法，1
'''
def findCornerByCenter(img):
    
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
    if len(contours)==0:
        print("Currently contours num is equal to Zero!")
        return img
    if len(contours)==1:
        img = cv2.drawContours(img,contours,-1,[255,255,0],2)
        return img
    if len(contours)==2:
        img = cv2.drawContours(img,contours,-1,[255,255,0],2)
        return img

    for idx,cnt in enumerate(contours):
        area = cv2.contourArea(cnt)
        if area<500:
            continue
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

    print("Contours Area(max 3):",area1,area2,area3)  
    for i in [idx1,idx2,idx3]:
        #计算轮廓中心点
        if i == -1:
            continue
        M=cv2.moments(contours[i])
        if M['m00']==0:
                M['m00']=0.001
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
    square_idx_list = []
    center_point_list = []
    for i in range(len(contours)):
        # 轮廓逼近
        epsilon = 0.01 * cv2.arcLength(contours[i], True)
        approx = cv2.approxPolyDP(contours[i], epsilon, True)
        # 分析几何形状
        corners = len(approx)
        shape_type = ""
        if corners == 4:
            M=cv2.moments(contours[i])
            if M['m00']==0:
                M['m00']=0.001
            
            cx = int(M['m10']/M['m00'])
            cy = int(M['m01']/M['m00'])
            img = cv2.circle(img,(cx,cy),1,[255,0,0],5)
            center_point = (cx,cy)
            center_point_list.append(center_point)
            square_idx_list.append(i)
    # 找到所有近似方形的轮廓，找到面积最大的看看 
    square_idx = -1
    area_max = 0      
    for idx in square_idx_list:
        if idx==idx1 or idx==idx2 or idx==idx3:
            continue
        area = cv2.contourArea(contours[i])
        if area > area_max:
            area_max = area
            square_idx = idx
    # 如果等于-1.说明没有找到小方块        
    if square_idx ==-1:
        print("can't find the square!")
    elif len(center_point_list)<=0:
        return img
    else:
        print("Square has found!,area:",cv2.contourArea(contours[square_idx]))
        M=cv2.moments(contours[square_idx])
        if M['m00']==0:
                M['m00']=0.001
        square_center_x = int(M['m10']/M['m00'])
        square_center_y = int(M['m01']/M['m00'])    
        square_center = (square_center_x,square_center_y)

        img = cv2.putText(img,"(%d,%d)"%(square_center_x,square_center_y),square_center,cv2.FONT_HERSHEY_PLAIN,0.5,(0,0,255))
        img = cv2.drawContours(img, contours,i,[255,255,0],2)
            

    return img



'''
根据颜色提取轮廓的角点检测算法，2
'''
def redContourExtract(img):
    time_start = time.time()
    img_origin = img.copy()
    img = cv2.cvtColor(img,cv2.COLOR_BGR2HSV)
    # range of red
    lower_red = np.array([150, 120, 20])
    upper_red = np.array([180, 180, 60])

    lower_red2 = np.array([0, 120, 20])
    upper_red2 = np.array([20, 180, 60])  # thers is two ranges of red

    mask_r = cv2.inRange(img, lower_red, upper_red)

    mask_r2 = cv2.inRange(img, lower_red2, upper_red2)

    mask = mask_r + mask_r2
    
    kernel = np.ones((3,3))
    img = cv2.dilate(mask,kernel)
    img = cv2.erode(img,kernel)
    # 这里已经可以得到四个角的很明显的图像。

    # 面积筛选
    center_point_list = []# 保存结果

    _,contours,hierarchy = cv2.findContours(img,cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    area1 = 0
    area2 = 0
    area3 = 0
    area4 = 0
    idx1 = -1
    idx2 = -1
    idx3 = -1
    idx4 = -1

    if len(contours)==0:
        print("Currently contours num is equal to Zero!")
        return img,center_point_list

    for idx,cnt in enumerate(contours):
        area = cv2.contourArea(cnt)
        minArea = 500 # 这个参数很重要
        if area<minArea:
            continue
        if area > area1:
            area4 = area3
            idx4 = idx3
            area3 = area2
            idx3 = idx2
            area2 = area1
            idx2 = idx1
            area1 = area
            idx1 = idx
        elif area > area2:
            area4 = area3
            idx4 = idx3
            area3 = area2
            idx3 = idx2
            area2 = area
            idx2 = idx
        elif area > area3:
            area4 = area3
            idx4 = idx3
            area3 = area
            idx3 = idx
        elif area > area4:
            area4 = area
            idx4 = idx

    
    for i in [idx1,idx2,idx3,idx4]:
        if i==-1:
            continue
        if cv2.contourArea(contours[i])==0:
            continue
        M=cv2.moments(contours[i])
        if M['m00']==0:
                M['m00']=0.001
        cx = int(M['m10']/M['m00'])
        cy = int(M['m01']/M['m00'])
        if i == idx1:
            print("the maxArea center point:",cx,cy)
        img_origin = cv2.circle(img_origin,(cx,cy),1,[255,0,255],5)
        center_point = (cx,cy)
        center_point_list.append(center_point)
        img_origin = cv2.putText(img_origin,"(%d,%d)"%(cx,cy),center_point,cv2.FONT_HERSHEY_PLAIN,2,(0,255,255),2)
        img_origin = cv2.drawContours(img_origin,contours,i,[255,0,255],3)

    # print("Area Max 4:",area1,area2,area3,area4)
    time_end = time.time()
    deltatime = time_end - time_start
    # print("one frame time:",deltatime)
    return img_origin,center_point_list



'''
借鉴apriltag算法的角点检测算法，3
'''
def apriltagFindCorner(frame):
    gray = np.array(cv2.cvtColor(frame,cv2.COLOR_RGB2GRAY))

    '''
    1.blur
    '''

    pass


'''
用PNP解算相机位置的算法
'''
def solvePoint(center_point_list):
    point_num = len(center_point_list)
    if point_num<4:
        print("Currently can't solve PnP problem!")
        return 

    camera_intrinsic_matrix = np.array(
        [[1557.90406532275,0,0],
        [0,1543.49983613257,0],
        [680.204169094410,644.295104426977,1]]).T
    
    # 这个矩阵这样写不知道对不对？
    camera_distortion = np.array(
        [-0.152787687060218,0.275768296110960,0,0,0]
    )

    '''
    World
    右下角方块角点：（0，0，0）

    右上角：（16，0，0）

    左下角：（0，21，0）

    左上角：（16，21，0）
    '''
    point_world1 = np.array([0,0,0],dtype=np.float32)
    point_world2 = np.array([16,0,0],dtype=np.float32)
    point_world3 = np.array([0,21,0],dtype=np.float32)
    point_world4 = np.array([16,21,0],dtype=np.float32)

    if point_num==4:
       '''
       这里放一个判断相对位置的算法
       '''
       point_world = np.stack([point_world1,point_world2,point_world3,point_world4]) # shape: 4*3
       point_image = np.stack([center_point_list[0],center_point_list[1],center_point_list[2],center_point_list[3]])
       # print(point_image, point_world)
       success, rotation_vector, translation_vector = cv2.solvePnP (point_world,point_image,camera_intrinsic_matrix,camera_distortion,flags=cv2.SOLVEPNP_ITERATIVE)
       # print("rotation", rotation)

       #这里借用一下公式
       rotM = cv2.Rodrigues(rotation_vector)[0]
       position = -np.matrix(rotM).T * np.matrix(translation_vector)
       return position



'''
通过鼠标获得hsv数值的自动调参算法
'''
def autoHSVget(img):
    
    img_HSV=cv2.cvtColor(img,cv2.COLOR_BGR2HSV)

    def getpos(event,x,y,flags,param):
        if event==cv2.EVENT_LBUTTONDOWN: #定义一个鼠标左键按下去的事件
            print(img_HSV[y,x])

    cv2.imshow("imageHSV",img_HSV)
    cv2.imshow('image',img)
    cv2.setMouseCallback("imageHSV",getpos)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


'''
使用mind vision相机进行角点检测，使用自撰算法，版本V1
'''
def CameraLoopCornerFinding1():
        DevList = mvsdk.CameraEnumerateDevice()
        nDev = len(DevList)
        if nDev < 1:
            print("No camera was found!")
            return

        for i, DevInfo in enumerate(DevList):
            print("{}: {} {}".format(i, DevInfo.GetFriendlyName(), DevInfo.GetPortType()))

        cams = []
        for i in map(lambda x: int(x), input("Select cameras: ").split()):
            cam = mvcamera.Camera(DevList[i])
            if cam.open():
                cams.append(cam)

        framecount = 0
        origin_time = time.time()
        time_all = 0
        while (cv2.waitKey(1) & 0xFF) != ord('q'):
            for cam in cams:
                frame = cam.grab()
                framecount+=1
                if frame is not None:
                    frame = cv2.resize(frame, (640,480), interpolation = cv2.INTER_LINEAR)
                    
                    
                    '''
                    输入frame处，在此对frame做修改
                    '''
                    frame_detection,centerpoint_list = redContourExtract(frame)

                    cv2.imshow("Vision Localization V2.0".format(cam.DevInfo.GetFriendlyName()), frame_detection)
                    current_time = time.time()
                    delta_time = current_time - origin_time
                    origin_time = current_time
                    time_all += delta_time

                    if framecount==100:#每1s计算一次
                        fps = framecount/time_all
                        time_all = 0
                        framecount = 0
                        print("current fps:", fps)

        for cam in cams:
            cam.close()


'''
使用apriltag算法，暂时没写好
'''
def CameraLoopCornerFinding2():
        DevList = mvsdk.CameraEnumerateDevice()
        nDev = len(DevList)
        if nDev < 1:
            print("No camera was found!")
            return

        for i, DevInfo in enumerate(DevList):
            print("{}: {} {}".format(i, DevInfo.GetFriendlyName(), DevInfo.GetPortType()))

        cams = []
        for i in map(lambda x: int(x), input("Select cameras: ").split()):
            cam = mvcamera.Camera(DevList[i])
            if cam.open():
                cams.append(cam)

        framecount = 0
        origin_time = time.time()
        time_all = 0

        
        while (cv2.waitKey(1) & 0xFF) != ord('q'):
            for cam in cams:
                frame = cam.grab()
                framecount+=1
                if frame is not None:
                    frame = cv2.resize(frame, (640,480), interpolation = cv2.INTER_LINEAR)
                    
                    
                    '''
                    输入frame处，在此对frame做修改
                    '''
                    frame = redContourExtract(frame)
                
                    cv2.imshow("Vision Localization V1.0".format(cam.DevInfo.GetFriendlyName()), frame)
                    current_time = time.time()
                    delta_time = current_time - origin_time
                    origin_time = current_time
                    time_all += delta_time

                    if framecount==100:#每100帧计算一次
                        fps = framecount/time_all
                        time_all = 0
                        framecount = 0
                        print("current fps:", fps)

        for cam in cams:
            cam.close()









parser = argparse.ArgumentParser(description='corner point detection')

parser.add_argument('--mode',dest='mode', type=str,required=True,
                    help='choose the mode, camera or img or test')
args = parser.parse_args()

if __name__=="__main__":
    mode = args.mode

    if mode == "img":
        img = cv2.imread("example2.jpg")
        autoHSVget(img)
        img = redContourExtract(img)
        
        cv2.imshow("img",img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    elif mode == "camera":
        try:
            print("camera mode beginning....")
            CameraLoopCornerFinding1()
        finally:
            cv2.destroyAllWindows()

    elif mode == "test":

        
        a = np.array([1,1],dtype=np.float32)
        b = np.array([17,1],dtype=np.float32)
        c = np.array([1,22],dtype=np.float32)
        d = np.array([17,22],dtype=np.float32)
        point_list = [a,b,c,d]
        position = solvePoint(point_list)
        print(position)
    else:
        print("Error:This mode is not available")