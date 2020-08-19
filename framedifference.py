import cv2
import os
import numpy as np
import processing

class FrameDifference:

    # PARAMETRES
    # video : frame sequence
    # noisek : gaussian blur kernel
    # blobk : dilatation kernel
    # it : dilatation iterations
    # noiseacti : Enable or Disable (TRUE / False) noise removing
    # False : Enable or Disable (TRUE / False) dilatation
    # k : jump

    def __init__(self, video, threshold, noisek, noisethresh, blobk, it, noiseacti, blobacti, k=1):
        self.video = video
        self.buffer = video[0]
        self.k = k
        self.threshold = threshold
        self.counter = 0
        self.noisk = noisek
        self.blobk = blobk
        self.it = it
        self.noisethresh = noisethresh
        self.noiseacti = noiseacti
        self.blobacti = blobacti



    def difference(self, f2):
        f = cv2.absdiff(self.buffer, f2)
        ret, diff = cv2.threshold(f, self.threshold, 255, cv2.THRESH_BINARY)
        if self.noiseacti:
            diff = processing.reduce_noise(diff, self.noisk, self.noisethresh)

        if self.blobacti:
            diff = processing.refine_blob(diff, self.blobk, self.it)
        
        return diff


    
    def start(self):
        for i in range(1, len(self.video)):
            frame = self.video[i]
            self.counter += 1
            
            diff = self.difference(frame)
            cv2.imshow('original', frame)
            cv2.imshow('substraction', diff)

            if cv2.waitKey(25) & 0xFF == ord('q'):
                break

            if self.counter == self.k:

                self.counter = 0
                self.buffer = frame

        cv2.destroyAllWindows()



DATAPATH = 'data/RGB_input/blizzard'
VIDEO = []

for di in sorted(os.listdir(DATAPATH), key=lambda s: int(s.split('.')[0])):
    VIDEO.append(cv2.imread(DATAPATH + '/' + di, 0))
    

DifferenceHandler = FrameDifference(VIDEO, 10, (7, 7), 150, (11, 11), 5, True, False, 1)
DifferenceHandler.start()