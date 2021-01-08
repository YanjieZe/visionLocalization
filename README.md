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

## 4 use mindVison camera for detection(haven't been finished)
> python detect.py --weights best.pt  --source 0 --view-img

# Process Log
**2020.11.13** start to write some bullshit

**2020.12.13** finish the RMset version 1.0 and use yolov5 to get over 80% accuracy

**2020.12.17** finish the collection module(grabFrame.py): **connect the mindvision camera and auto collect frame as the raw dataset**

**2021.01.08** start to modify the code but then decide to write **corner detection** first
开始自己尝试写角点检测算法。第一个想法是先把他切割出来，就是用yolo分割的结果做一个切割。简单的**getCropped()**实现一下。
第二个想法是用这个tag最明显的位置来进行角点的推算，有两个很容易检测的位置：中间的字母，右下角的一个小方块。那么我就先把轮廓提取出来，再把这两个的轮廓拿出来，求一下中心点的位置，就可以得到描述这个tag的两个特征。先把这个想法实现一下吧。得到这两个中心点的坐标后tag的四个角点坐标基本就出来了。

把轮廓面积排了一下，莫名和预计中的不一样。
# Bug log
## 1."bort" (this may appear on MacOS)
solution: 
> sudo code .