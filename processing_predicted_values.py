#!/usr/bin/env python
# coding: utf-8

import cv2
import numpy as np
import os
import yaml
from yaml.loader import SafeLoader

class YOLO_Pred():
    def __init__(self,onnx_model,data_yaml):
        # Loading yaml file
        with open(data_yaml,mode = 'r') as f:
            data_yaml = yaml.load(f,Loader = SafeLoader)
        self.labels = data_yaml['names']
        self.nc = data_yaml['nc']
        # Loading YOLO Model
        self.yolo = cv2.dnn.readNetFromONNX(onnx_model)
        self.yolo.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
        self.yolo.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)

    def predictions(self,image):

        row, col, d = image.shape
        # converting image into square image
        max_rc = max(row,col)
        input_image = np.zeros((max_rc,max_rc,3),dtype = np.uint8)
        input_image[0:row,0:col] = image
        #getting prediction
        input_wh = 640
        blob = cv2.dnn.blobFromImage(input_image,1/255,(input_wh,input_wh),swapRB=True,crop=False)
        self.yolo.setInput(blob)
        preds = self.yolo.forward()
        # Filtering using confidence (0.4) and probability score (.25)
        detections = preds[0]
        boxes = []
        confidences =[]
        classes = []
        image_w, image_h = input_image.shape[0:2]
        x_factor = image_w/input_wh
        y_factor = image_h/input_wh
        for i in range(len(detections)):
            row = detections[i]
            confidence = row[4] # confidence of detecting an object
            if confidence >= 0.4:
                class_score = row[5:].max()  # maximum probability object
                class_id = row[5:].argmax()
                if class_score >= 0.25:
                    cx, cy, w, h = row[0:4] # getting the centre_x,centre_y,w,h of bounding box
                    # constructing the bounding box
                    left = int((cx - 0.5*w)*x_factor)  
                    top = int((cy - 0.5*h)*y_factor)
                    width = int(w*x_factor)
                    height = int(h*y_factor)
                    
                    box = np.array([left,top,width,height])
        
                    confidences.append(confidence)
                    boxes.append(box)
                    classes.append(class_id)
        
        boxes = np.array(boxes).tolist()
        confidences = np.array(confidences).tolist()
        
        # Non Maximum Suppression
        index = cv2.dnn.NMSBoxes(boxes,confidences,0.25,0.45).flatten()
        
        # Drawing the boxes
        for ind in index:
            x,y,w,h = boxes[ind]
            bb_conf = np.round(confidences[ind],2)
            classes_id = classes[ind]
            class_name = self.labels[classes_id]
            colours = self.generate_colours(classes_id)
            text = f'{class_name}: {bb_conf}'
            cv2.rectangle(image,(x,y),(x+w,y+h),colours,2)
            cv2.rectangle(image,(x,y-30),(x+w,y),colours,-1)
            cv2.putText(image,text,(x,y-10),cv2.FONT_HERSHEY_PLAIN,0.7,(0,0,0),1)
        
        return image

    def generate_colours(self,ID):
        np.random.seed(10)
        colours = np.random.randint(100,255,size = (self.nc,3)).tolist()
        return tuple(colours[ID])
