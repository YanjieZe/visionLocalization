import argparse
import time
from pathlib import Path

import cv2
import torch
import torch.backends.cudnn as cudnn
from numpy import random

from models.experimental import attempt_load
from utils.datasets import LoadStreams, LoadImages
from utils.general import check_img_size, non_max_suppression, apply_classifier, scale_coords, xyxy2xywh, \
    strip_optimizer, set_logging, increment_path
from utils.plots import plot_one_box
from utils.torch_utils import select_device, load_classifier, time_synchronized
import mvsdk
import mvcamera
from cornerfinding.cornerfinding import *

def detect(weights,imgsz=640,_device=''):
    view_img = True
    cudnn.benchmark = True
     # Load model
    device = select_device(_device)  
    model = attempt_load(weights, map_location=device)  # load FP32 model
    imgsz = check_img_size(imgsz, s=model.stride.max())  # check img_size
    half = device.type != 'cpu'  # half precision only supported on CUDA
    if half:
        model.half()  # to FP16
    # 打开相机
    DevList = mvsdk.CameraEnumerateDevice()
    nDev = len(DevList)

    if nDev < 1:
            print("No camera was found!")
    for j, DevInfo in enumerate(DevList):
            print("{}: {} {}".format(j, DevInfo.GetFriendlyName(), DevInfo.GetPortType()))

    cams = []
    for j in map(lambda x: int(x), input("Select cameras: ").split()):
            cam = mvcamera.Camera(DevList[j])
    if cam.open():
            cams.append(cam)

    cap = cams[0]

    dataset = LoadStreams(0,img_size=imgsz,camera=cap)

    # Get names and colors
    names = model.module.names if hasattr(model, 'module') else model.names
    colors = [[random.randint(0, 255) for _ in range(3)] for _ in names]

    # Run inference
    t0 = time.time()

    img = torch.zeros((1, 3, imgsz, imgsz), device=device)  # init img
    _ = model(img.half() if half else img) if device.type != 'cpu' else None  # run once
    
    
    for path, img, im0s, vid_cap in dataset:
        img_origin = img.copy()
        img = torch.from_numpy(img).to(device)
        img = img.half() if half else img.float()  # uint8 to fp16/32
        img /= 255.0  # 0 - 255 to 0.0 - 1.0
        if img.ndimension() == 3:
            img = img.unsqueeze(0)

        # Inference
        t1 = time_synchronized()
        pred = model(img, augment=True)[0]
        
        # Apply NMS
        pred = non_max_suppression(pred, conf_thres, iou_thres, classes, agnostic=agnostic_nms)
        t2 = time_synchronized()

        # # Process detections
        # for i, det in enumerate(pred):  # detections per image
            
        #     _, s, im0 = _, '%g: ' % i, im0s[i].copy()
    
           
        #     s += '%gx%g ' % img.shape[2:]  # print string
        #     gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]  # normalization gain whwh
        #     if len(det):
        #         # Rescale boxes from img_size to im0 size
        #         det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0.shape).round()

        #         # Print results
        #         for c in det[:, -1].unique():
        #             n = (det[:, -1] == c).sum()  # detections per class
        #             s += '%g %ss, ' % (n, names[int(c)])  # add to string

        #         # Write results
        #         for *xyxy, conf, cls in reversed(det):
     
        #             if view_img:  # Add bbox to image
        #                 label = '%s %.2f' % (names[int(cls)], conf)
        #                 plot_one_box(xyxy, im0, label=label, color=colors[int(cls)], line_thickness=3)

        #     # Print time (inference + NMS)
        #     print('%sDone. (%.3fs)' % (s, t2 - t1))

        #     if view_img:
        #         cv2.imshow("prediction result", im0)
        #         if cv2.waitKey(1) == ord('q'):  # q to quit
        #             cam.close()
        #             raise StopIteration
        
        '''
        对detection结果做处理
        '''
        frame = img_origin
        for i,det in enumerate(pred):
            if det.shape[0]<=0 or det.shape[1]<6 or det[0][4]<=0.001:
                continue
            img_cropped = getCropped(img_origin,det)
            img_contour,centerpoint_list = redContourExtract(img_cropped)

            position = solvePoint(center_point_list=centerpoint_list)
            if isinstance(position,int)!=1:
                frame = cv2.putText(img_contour,"camera postion:(%d,%d,%d)"%(position[0],position[1],position[2]),(20,20),cv2.FONT_HERSHEY_PLAIN,2,(255,255,0),2)          
        cv2.imshow("img",frame)    
        if cv2.waitKey(1) == ord('q'):
                cap.close()
                raise StopIteration
        

if __name__=="__main__":
    conf_thres = 0.25
    iou_thres =  0.45
    classes = 0
    agnostic_nms = True

    weights = "best.pt"
    detect(weights)