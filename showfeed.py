import numpy as np
import cv2
import imutils
import time
import device_config as CONFIG
from imutils.video import WebcamVideoStream
from imutils.video.pivideostream import PiVideoStream
from imutils.video import VideoStream
from imutils.video import FPS

def rot90(img, rotflag):
        #""" rotFlag 1=CW, 2=CCW, 3=180"""
    if rotflag == 1:
        img = cv2.transpose(img)
        img = cv2.flip(img, 1)  # transpose+flip(1)=CW
    elif rotflag == 2:
        img = cv2.transpose(img)
        img = cv2.flip(img, 0)  # transpose+flip(0)=CCW
    elif rotflag ==3:
        img = cv2.flip(img, -1)  # transpose+flip(-1)=180
    elif rotflag != 0:  # if not 0,1,2,3
        raise Exception("Unknown rotation flag({})".format(rotflag))
    return img


#vs = WebcamVideoStream(src=0).start()
#vs = VideoStream(src=1).start()
vs = PiVideoStream().start()
time.sleep(2.0)
fps = FPS().start()

while(True):
    frame = vs.read()
    if CONFIG.CAM_ROT:
        frame = rot90(frame,CONFIG.ROT_FLAG)

    cv2.imshow('frame',frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        print("break")
        break
# Release everything if job is finishedvs.stop()
vs.stop()
cv2.destroyAllWindows()
