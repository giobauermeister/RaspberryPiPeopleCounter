import numpy as np
import cv2
import imutils
import time
import datetime
import argparse
import device_config as CONFIG
from imutils.video import FPS
from imutils.video.pivideostream import PiVideoStream

def rot90(img, rotflag):
    # """ rotFlag 1=CW, 2=CCW, 3=180"""
    if rotflag == 1:
        img = cv2.transpose(img)
        img = cv2.flip(img, 1)  # transpose+flip(1)=CW
    elif rotflag == 2:
        img = cv2.transpose(img)
        img = cv2.flip(img, 0)  # transpose+flip(0)=CCW
    elif rotflag == 3:
        img = cv2.flip(img, -1)  # transpose+flip(-1)=180
    elif rotflag != 0:  # if not 0,1,2,3
        raise Exception("Unknown rotation flag({})".format(rotflag))
    return img

vs = PiVideoStream().start()
time.sleep(2.0)

for num in range(1,2):
    frame = vs.read()
    frame = imutils.resize(frame,width=400)
    if CONFIG.CAM_ROT:
        frame = rot90(frame,CONFIG.ROT_FLAG)

    cv2.line(frame, (0, int(frame.shape[0] * CONFIG.LOWER_LINE_HEIGHT)),
             (frame.shape[1], int(frame.shape[0] * CONFIG.LOWER_LINE_HEIGHT)), (0, 255, 0), 3)
    cv2.line(frame, (0, int(frame.shape[0] * CONFIG.UPPER_LINE_HEIGHT)),
             (frame.shape[1], int(frame.shape[0] * CONFIG.UPPER_LINE_HEIGHT)), (0, 255, 0), 3)

    cv2.imwrite("img_"+str(num)+ ".jpg",frame)
    time.sleep(1.0)

cv2.destroyAllWindows()
vs.stop()