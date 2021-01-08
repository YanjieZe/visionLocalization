import cv2


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
        img_final = cv2.drawContours(img, contours,i,[255,255,0],2)
    print(map(cv2.contourArea,contours))
    # img_final = img_canny
    return img_final


if __name__=="__main__":
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


