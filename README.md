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

# Bug log
## 1."bort" (this may appear on MacOS)
solution: 
> sudo code .