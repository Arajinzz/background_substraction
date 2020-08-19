import cv2
import numpy as np


def reduce_noise(frame, kernel, threshold):
    frame = cv2.GaussianBlur(frame, kernel, 0)
    ret, frame = cv2.threshold(frame, threshold, 255, cv2.THRESH_BINARY)
    return frame


def refine_blob(frame, kernel, it):
    
    kernel1 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,kernel)
    frame = cv2.dilate(frame, kernel1, iterations=it)
    erosion = cv2.erode(frame, kernel1, iterations=it)
    #kernel1 = np.ones((5, 5), np.uint8)
    #frame = cv2.morphologyEx(frame, cv2.MORPH_CLOSE, kernel1)
    
    return frame