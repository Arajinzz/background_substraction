import cv2
import os
import numpy as np


class Single_Gaussian:
    def __init__(self, video, alpha, threshold):
        self.video = video
        self.alpha = alpha
        self.mean = self.video[0]
        self.h, self.w = self.mean.shape
        self.threshold = threshold
        self.var = np.ones((self.h, self.w), np.uint8)

    
    def start(self):
        for frame in VIDEO:
            new_mean = ((1-self.alpha) * self.mean + self.alpha * frame).astype(np.uint8)
            new_var = self.alpha * cv2.subtract(frame, self.mean)**2 + (1-self.alpha) * self.var

            val = cv2.absdiff(frame, self.mean) ** 0.5
            self.mean = np.where(val < self.threshold, new_mean, self.mean)
            self.var = np.where(val < self.threshold, new_var, self.var)

            bg = np.where(val < self.threshold, frame, 0)
            fg = np.where(val >= self.threshold, frame, 0)
            
            cv2.imshow('background', bg)
            cv2.imshow('forground', fg)
            
            if cv2.waitKey(25) & 0xFF == ord('q'):
                break

        cv2.destroyAllWindows()



DATAPATH = 'data/RGB_input/blizzard'
VIDEO = []

for di in sorted(os.listdir(DATAPATH), key=lambda s: int(s.split('.')[0])):
    VIDEO.append(cv2.imread(DATAPATH + '/' + di, 0))
    

Gau = Single_Gaussian(VIDEO, 0.1, 4)
Gau.start()