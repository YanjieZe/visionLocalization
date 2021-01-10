# CornerPointDetection
YanjieZe(the Last Refugee)
# Use method

## 1 use web cam for detection
old version
> python detection.py --mode webcam

new version
> python detect.py --weights best.pt  --source 0 --view-img

## 2 Collect frame as the raw dataset, using mindvision camera
> python grabFrame.py --collection 1 --reponum (the numbe your want) --framenum (the number you want)

## 3 Use mindVision camera,just for a look
> python grabFrame.py --collection 0

## 4 use mindVison camera for detection
this python file is in another folder.So you should first:
> cd cornerfinding
then:
> python cornerfinding.py

# Process Log
**2020.11.13** start to write some bullshit

**2020.12.13** finish the RMset version 1.0 and use yolov5 to get over 80% accuracy

**2020.12.17** finish the collection module(grabFrame.py): **connect the mindvision camera and auto collect frame as the raw dataset**

**2021.01.08** start to modify the code but then decide to write **corner detection** first
开始自己尝试写角点检测算法。第一个想法是先把他切割出来，就是用yolo分割的结果做一个切割。简单的**getCropped()**实现一下。
第二个想法是用这个tag最明显的位置来进行角点的推算，有两个很容易检测的位置：中间的字母，右下角的一个小方块。那么我就先把轮廓提取出来，再把这两个的轮廓拿出来，求一下中心点的位置，就可以得到描述这个tag的两个特征。先把这个想法实现一下吧。得到这两个中心点的坐标后tag的四个角点坐标基本就出来了。

把轮廓面积排了一下，莫名和预计中的不一样。但是可以确定的是，可以用面积筛选方法得到中间字符的位置，不过要得到右下角小方块的位置用面积筛选的方法不怎么具有鲁棒性，因此找小方块另寻方法。

第三个想法：小方块是方形的，可以用一种匹配形状的方法把他找到吗？这个想法让我使用检测形状的方法来找出小方块，即轮廓逼近法。写了一下，效果还可以，不过结果竟然有两个，其中一个很奇怪，是边缘的一个角点，应该是有一个很小的矩形轮廓，应该是误判了。

因此第四个想法，再进一步优化一下，将轮廓逼近法之后再用面积比较法筛选一下试一试。

**2021.01.09** continuing modify
昨天的第四个想法实现之后，开始用mind vision camera进行实时检测。
发现在实时检测之中，有可能出现找不到矩形框的情况。做了一些优化。
目前尚存在的问题：可能会检测出很小的轮廓。
考虑在面积筛选的时候做一些优化。

做完优化后，可以找到点，但是会有很多点在飘，接下来考虑要么使用目标跟踪的算法，要么重新优化原来的算法。
在新的图片中我发现tag的红色十分明显，考虑从这一步入手先试一试。原先的**findCorner**可以先不改，对于轮廓提取做出改进就行。
第五个想法，颜色分割+轮廓识别重调。

颜色分割的结果意外地好。开始进行轮廓的识别和位置的计算。

面积筛选和中心点计算已完成，效果很好！详情见**redContourExtract**
![](localizationV1.gif)

接下来就是增强鲁棒性。
1. 原图高曝光，红色更容易识别，但是在不同场景下会失效。考虑如何提高这一方面的颜色适应性。
2. 算法中的颜色区域可能不够精准，考虑通过实时返回的hsv值对要识别的区域进行标定。

第六个想法，通过实时获得hsv值对想要得到的区域进行标定，再进行检测，即部分自动参数设置功能。

写好了一个取参数的函数**autoHSVget**

# Bug log
## 1."bort" (this may appear on MacOS)
solution: 
> sudo code .