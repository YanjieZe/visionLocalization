# CornerPointDetection
YanjieZe(the Last Refugee)
# Use method
## 1 use web cam to detection
enter this line of code in your command line to use the wec camera mode.

> python detection.py --mode webcam

## 2 Collect frame as the raw dataset, using mindvision camera
> python cv_grab.py --collection 1 --reponum (the numbe your want) --framenum (the number you want)

## 3 Use mindVision camera,just for a look
> python cv_grab.py --collection 0

# Process Log
**11.13** start to write some bullshit

**12.13** finish the RMset version 1.0 and use yolov5 to get over 80% accuracy

**12.17** finish the collection module: **connect the mindvision camera and auto collect frame as the raw dataset**


# Bug log
## 1."bort" (this may appear on MacOS)
solution: 
> sudo code .