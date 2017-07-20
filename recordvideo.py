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


ap = argparse.ArgumentParser()
ap.add_argument("-v", "--video", action='store_true', default=False,
	help="show video feed")
args = vars(ap.parse_args())

with_video= args["video"]

vs = PiVideoStream().start()
time.sleep(2.0)

# Define the codec and create VideoWriter object
fourcc = cv2.VideoWriter_fourcc(*'XVID')
#out = cv2.VideoWriter('output.avi',fourcc, 20.0, (640,480))
writer = None
start_thread = datetime.datetime.now()
while(True):
    frame = vs.read()
    frame = imutils.resize(frame,width=400)
    if CONFIG.CAM_ROT:
        frame = rot90(frame,CONFIG.ROT_FLAG)
    #frame = rot90(frame,2)

    #frame = cv2.flip(frame,180)

    if writer is None:
        writer = cv2.VideoWriter('output_tst.avi', fourcc, 30.0, (frame.shape[1],frame.shape[0]))
    # write the flipped frame
    writer.write(frame)

    if with_video:
        cv2.imshow('frame',frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        print("break")
        break

    seconds_elapsed = (datetime.datetime.now() - start_thread).seconds
    if seconds_elapsed > 30:
        print("30 seconds recorded stopping now")
        break

# Release everything if job is finishedvs.stop()
cv2.destroyAllWindows()
vs.stop()
writer.release()

