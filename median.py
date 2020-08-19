import cv2
import os
import numpy as np

class Median:

    # PARAMETRES 
    # video : frame sequence
    # N : frame buffer size

    def __init__(self, video, threshold, N=1):
        self.video = video
        self.buffer = video[0].flatten()
        self.buffer = self.buffer.reshape(-1, self.buffer.shape[0])
        self.N = N
        self.threshold = threshold
        self.counter = 0

    

    def median(self, frame):
        
        h, w = frame.shape
        flat_arr = frame.flatten()
        median_frame = self.buffer[int((self.buffer.shape[0] - 1) // 2)]
        med_frame = cv2.absdiff(flat_arr, median_frame)
        ret, med_frame = cv2.threshold(med_frame, self.threshold, 255, cv2.THRESH_BINARY)
        med_frame = med_frame.reshape(h, w)
        flat_arr = flat_arr.reshape(-1, flat_arr.shape[0])

        #print(self.buffer.shape)

        if(self.N == 1):
            self.buffer = flat_arr
            return med_frame
        
        if(self.buffer.shape[0] == self.N):
            self.buffer = self.buffer[1:]

        # add to buffer
        self.buffer = np.append(self.buffer, flat_arr, axis=0)
        # sort buffer

        self.buffer = np.sort(self.buffer, axis=0)
        return med_frame


    def start(self):
        for i in range(1, len(self.video)):
            frame = self.video[i]

            diff = self.median(frame)

            cv2.imshow('original', frame)
            cv2.imshow('median', diff)

            if cv2.waitKey(25) & 0xFF == ord('q'):
                break
        
        cv2.destroyAllWindows()



DATAPATH = 'data/RGB_input/blizzard'
VIDEO = []

for di in sorted(os.listdir(DATAPATH), key=lambda s: int(s.split('.')[0])):
    VIDEO.append(cv2.imread(DATAPATH + '/' + di, 0))
    
# 25 : threshold
# 10 : frame buffer size
DifferenceHandler = Median(VIDEO, 25, 10)
DifferenceHandler.start()