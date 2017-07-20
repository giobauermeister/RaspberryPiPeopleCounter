# import the necessary packages
from __future__ import print_function
import numpy as np
import argparse
import distutils.util
from collections import deque
import datetime
import json
import imutils
import cv2
import time
import pt_config2
import sys, getopt
import requests
from threading import Thread
import blobs2
import kalman_munkres_object_tracker
from imutils.video import FPS

global sending

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

def post():
    global sending
    payload = {"metadata":
        {
            "total in": number_of_ppl_in,
            "total out": number_of_ppl_out
        },
        "workflowid": mr_wf_id}
    headers = {'content-type': 'application/json','accept':'application/json', 'api-key':post_key}
    response = requests.post(post_url, data=json.dumps(payload), headers=headers)
    sending = False

def hellomr():
    global sending
    payload = {"metadata":
        {
            "online":"Hi im online"
        },
        "workflowid": mr_wf_id}
    headers = {'content-type': 'application/json','accept':'application/json', 'api-key':post_key}
    response = requests.post(post_url, data=json.dumps(payload), headers=headers)
    sending = False

# Construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-v", "--video", action='store_true', default=False,
	help="show video feed")
ap.add_argument("-t", "--trackbars", action='store_true', default=False,
	help="trackbars for adjustment")
ap.add_argument("-a", "--analyse", action='store_true', default=False,
	help="eroded dilated view")
ap.add_argument("-r", "--recorded", action='store_false', default=True,
	help="playback a recorded video")
args = vars(ap.parse_args())

# Setup
pts = deque(maxlen=64)
srcTest = 'output_tst_top.avi'
#srcTest = 'output_tst_rec.avi'
#srcTest = 'output_tst_big_room.avi'
#srcTest = 'output_tst_kitchen_2.avi'
#srcTest = 'output_tst_middle.avi'
#srcTest = 'output_tst_1.avi'
savevideo=False
log_enabled = False
analyse_from_stream = args["recorded"]
with_video= args["video"]
with_analyse = args["analyse"]
with_trackbar = args["trackbars"]

# Mobile response
post_to_mr = True
post_url = 'https://rest.mobileresponse.io/1/workflows'
post_key = 'c5beaf5d-cc32-4f30-b139-a048bedc4750'
mr_wf_id = "7a832bec-dbb6-4b50-89fd-a319a25f8e4f"

# main thread timer
timer_enabled = False

# Settings
# move to config
# Middle room camera = 2000
min_contour = 1000
acc_w = 0.3

#lineheight
upper_lineheight = 0.6
lower_lineheight = 0.4

#dist to line
dist_line = 10

# top er
er_w = 4
er_h = 4
di_w = 15
di_h = 20

# trackbar
if with_trackbar:
    def nothing(x):
        pass
    cv2.namedWindow("controls")
    cv2.createTrackbar('t_er_w','controls',er_w,25,nothing)
    cv2.createTrackbar('t_er_h','controls',er_h,25,nothing)
    cv2.createTrackbar('t_di_w','controls',di_w,25,nothing)
    cv2.createTrackbar('t_di_h','controls',di_h,25,nothing)
    cv2.createTrackbar('contour_area','controls',min_contour,10000,nothing)

# Tracker
tracker = blobs2.BlobTracker()
km_tracker = kalman_munkres_object_tracker.BlobTracker()
# Init variables
out = 0
vs = 0
cap = 0
number_of_ppl_out = 0
number_of_ppl_in = 0
current_frame = 0
frame = 0
sending = False

# fps
fps = FPS().start()

# Dont know
switch = '0 : PASS \n1 : STOP'

if savevideo:
    out = cv2.VideoWriter('output.avi', -1, 30.0, (640, 480))

# Load video or stream
if analyse_from_stream:
    from imutils.video.pivideostream import PiVideoStream
    #vs = WebcamVideoStream(src=0).start()
    vs = PiVideoStream().start()
    time.sleep(2.0)
else:
    cap = cv2.VideoCapture(srcTest)
    cap.set(cv2.CAP_PROP_FPS, 30)


# For background subtraction
masks = pt_config2.masks
for_er = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (pt_config2.er_w, pt_config2.er_h))
for_di = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (pt_config2.di_w, pt_config2.di_h))

# Calibrate algorithm for stream or video
if analyse_from_stream:
    frame = vs.read()
    frame = imutils.resize(frame, width=400)
    #frame = imutils.resize(frame, width=240)
    #frame = rot90(frame,2)
    c_zero = np.float32(frame)
    #vs.stream.set(0, 000.0)
    width = int(frame.shape[1])
    height = int(frame.shape[0])
    fourcc = "mp4v"
    frames = 1

else:
    _, frame = cap.read()
    frame = imutils.resize(frame, width=240)
    c_zero = np.float32(frame)
    # cap.set(0, 000.0)
    width = int(frame.shape[1])
    height = int(frame.shape[0])
    fourcc = "mp4v"
    # mjpg h624
    frames = 1

# Init stuff for the tracker
tracker.FRAME = frame.copy()
km_tracker.FRAME = frame.copy()
trails = np.zeros((height, width, 3)).astype(np.uint8)
km_trails = np.zeros((height, width, 3)).astype(np.uint8)
start_t = time.clock()

# welcome call
sending = True
hellomr()

# TODO: check if total line crossed the two lines + direction to calculate if it crossed or not + dx, dy between most outer coordinates
# Main loop where the magic happens
def crossed_line(v, frame, upper_lineheight, lower_lineheight, dist_line, var_out, var_in, sending):
    # check if it crossed the line
    # TODO: check by checking if the trail did
    if (abs(v.y - (frame.shape[0] * upper_lineheight)) < dist_line):
        if with_video:
            cv2.line(frame, (0, int(frame.shape[0] * upper_lineheight)),
                     (frame.shape[1], int(frame.shape[0] * upper_lineheight)),
                     (0, 0, 255), 3)
        # crossed upper with a negative speed
        if (v.dy > 0 and v.crossed_lower and not v.crossed_upper):
            var_out += 1
            v.crossed_upper = True
            v.counted = True
        elif (v.dy > 0 and v.counted):
            v.crossed_upper = False
            v.crossed_lower = False
            v.counted = False
        else:
            v.crossed_upper = True
            # if True:
            #   cv2.imwrite("passing_out_"+str(number_of_out_in)+".jpg",frame)
        if post_to_mr and not sending:
            sending = True
            # detect people in the image
            start_thread = datetime.datetime.now()
            t = Thread(target=post)
            t.daemon = True
            t.start()
            if log_enabled:
                print("[INFO] deploying send thread took: {}s".format(
                    (datetime.datetime.now() - start_thread).total_seconds()))
                # MRPoster.do_post(number_of_ppl_in,number_of_ppl_out)
                # cv2.imshow('ROI', frame)
                # cv2.waitKey(15)
    if (abs(v.y - (frame.shape[0] * lower_lineheight)) < dist_line):
        if with_video:
            cv2.line(frame, (0, int(frame.shape[0] * lower_lineheight)),
                     (frame.shape[1], int(frame.shape[0] * lower_lineheight)),
                     (0, 0, 255), 3)
        # check direction
        if (v.dy < 0 and v.crossed_upper and not v.crossed_lower):
            v.crossed_lower = True
            v.counted = True
            var_in += 1
            # if True:
            #    cv2.imwrite("passing_out_"+str(number_of_ppl_out)+".jpg",frame)
        elif (v.dy < 0 and v.counted):
            v.crossed_upper = False
            v.crossed_lower = False
            v.counted = False
        else:
            v.crossed_lower = True
        if post_to_mr and not sending:
            sending = True
            start_thread = datetime.datetime.now()
            # Post to mobileresponse in a thread
            t = Thread(target=post)
            t.daemon = True
            t.start()
            if log_enabled:
                print("[INFO] deploying send thread took: {}s".format(
                    (datetime.datetime.now() - start_thread).total_seconds()))
                # anti bounce
    return var_in, var_out, sending, frame

def line(p1, p2):
    A = (p1[1] - p2[1])
    B = (p2[0] - p1[0])
    C = (p1[0]*p2[1] - p2[0]*p1[1])
    return A, B, -C

def intersection(L1, L2):
    D  = L1[0] * L2[1] - L1[1] * L2[0]
    Dx = L1[2] * L2[1] - L1[1] * L2[2]
    Dy = L1[0] * L2[2] - L1[2] * L2[0]
    if D != 0:
        x = Dx / D
        y = Dy / D
        return (x,y)
    else:
        return False


T = np.array([[0, -1], [1, 0]])

def line_intersect(a1, a2, b1, b2):
    da = np.atleast_2d(a2 - a1)
    db = np.atleast_2d(b2 - b1)
    dp = np.atleast_2d(a1 - b1)
    dap = np.dot(da, T)
    denom = np.sum(dap * db, axis=1)
    num = np.sum(dap * dp, axis=1)
    return np.atleast_2d(num / denom).T * db + b1


def trail_crossed_lines(dead_blobs, frame, upper_lineheight, lower_lineheight, var_out, var_in, sending):

    for v in dead_blobs:

        # loop through trail
        p_a = v.traces[0]
        p_b = v.traces[-1]

        cv2.line(frame, (p_a[0],p_a[1]), (p_b[0],p_b[1]), (255,0,255), 5)

        #  lines to cross
        l_a = (0, int(frame.shape[0] * upper_lineheight))
        l_b = (frame.shape[1], int(frame.shape[0] * upper_lineheight))

        cv2.line(frame, (l_a[0],l_a[1]), (l_b[0],l_b[1]), (0,255,255), 5)

        cv2.imshow('ROI', frame)
        cv2.waitKey(2000)

        intersect = line_intersect(p_a[0],p_b,l_a,l_b)
        print("i: ",intersect)



        if R:
            print("Intersection detected:", R)
            cv2.circle(frame, R, 5, (255, 0, 255), -1)
            cv2.waitKey(20)
        else:
            print("No single intersection point detected")




def update_trails(arg_tracker, arg_trails, color):
    c = color
    for v in arg_tracker.virtual_blobs:

        ox = None
        oy = None

        # for i in range(len(v.pred) - 1): cv2.line(frame, v.pred[i], v.pred[i + 1], v.color)
        if len(v.traces) > 2:
            #print(len(arg_tracker.traces[v.id]))
            for pos in v.traces:
                x = int(pos[0])
                y = int(pos[1])

                if ox and oy:
                    sx = int(0.8 * ox + 0.2 * x)
                    sy = int(0.8 * oy + 0.2 * y)

                    #  Colours are BGRA
                    if c is None:
                        c = v.color
                    cv2.line(arg_trails, (sx, sy), (ox, oy), c, 1)
                    # cv2.line(alpha, (sx, sy), (ox, oy), (255,255,255), 2)
                    oy = sy
                    ox = sx
                else:
                    ox, oy = x, y
    return arg_trails

def update_trails_old(arg_tracker, arg_trails, color):
    c = color
    for v in arg_tracker.virtual_blobs:

        ox = None
        oy = None

        # for i in range(len(v.pred) - 1): cv2.line(frame, v.pred[i], v.pred[i + 1], v.color)
        if len(arg_tracker.traces) > 2:
            #print(len(arg_tracker.traces[v.id]))
            for pos in arg_tracker.traces[v.id]:
                x = int(pos[0])
                y = int(pos[1])

                if ox and oy:
                    sx = int(0.8 * ox + 0.2 * x)
                    sy = int(0.8 * oy + 0.2 * y)

                    #  Colours are BGRA
                    if c is None:
                        c = v.color
                    cv2.line(arg_trails, (sx, sy), (ox, oy), c, 1)
                    # cv2.line(alpha, (sx, sy), (ox, oy), (255,255,255), 2)
                    oy = sy
                    ox = sx
                else:
                    ox, oy = x, y
    return arg_trails



while True:
    cv2.waitKey(30)
    trails = np.zeros((height, width, 3)).astype(np.uint8)
    km_trails = np.zeros((height, width, 3)).astype(np.uint8)
    # with a pause here the feed can replay slower


    # to check the time an execution of the whole mainloop takes
    if timer_enabled:
        start = datetime.datetime.now()

    # detect people in the image from stream or video
    if analyse_from_stream:
        frame = vs.read()
        #frame = imutils.resize(frame,width=240)
        frame = imutils.resize(frame,width=400)
      #  frame = rot90(frame,2)
    else:
        _, frame = cap.read()
        frame = imutils.resize(frame, width=240)
       # frame = imutils.resize(frame, width=400)

    # Increment the current frame counter
    current_frame +=1

    # do work
    if with_trackbar:
        for_er = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (cv2.getTrackbarPos('t_er_w','controls'), cv2.getTrackbarPos('t_er_h','controls')))
        for_di = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (cv2.getTrackbarPos('t_di_w','controls'), cv2.getTrackbarPos('t_di_h','controls')))

    # Subtraction
    cv2.accumulateWeighted(frame, c_zero, acc_w)

    # im_zero = cv2.convertScaleAbs(c_zero)
    im_zero = c_zero.astype(np.uint8)

    #  Get the first diff image - this is raw motion
    d1 = cv2.absdiff(frame, im_zero)

    #  Convert this to greyscale
    gray_image = cv2.cvtColor(d1, cv2.COLOR_BGR2GRAY)

    #  ksize aperture linear size, must be odd
    # gray_smooth = cv2.medianBlur(gray_image, 5)

    #  Turn this into a black and white image (white is movement)
    thresh, im_bw = cv2.threshold(gray_image, 15, 255, cv2.THRESH_BINARY)

    #  Erode and Dilate Image to make blobs clearer.  Adjust erosion and dilation values in pt_config
    im_er = cv2.erode(im_bw, for_er)
    im_dl = cv2.dilate(im_er, for_di)

    # mask out ellipitical regions
    for mask in masks:
        cv2.ellipse(im_dl, (mask[0], mask[1]), (mask[2], mask[3]), 0, 0, 360, (0, 0, 0), -1)

    _, contours, hierarchy = cv2.findContours(im_dl, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Get trackbar value
    if with_trackbar:
        min_contour = cv2.getTrackbarPos('contour_area','controls')

    # Show analytic view
    if with_analyse:
        cv2.imshow("morph", frame)
        cv2.imshow("Im_zero", im_zero)
        cv2.imshow("Threshholded Image", im_bw)
        cv2.imshow('erode', im_er)
        cv2.imshow('dilate', im_dl)
        cv2.imshow("Eroded/Dilated Image", im_dl)

    my_blobs = []
    # Process targets
    for c in contours:
        # if the contour is too small, ignore it
        if cv2.contourArea(c) < min_contour:
            continue
        try:
            # compute the bounding box for the contour, draw it on the frame,
            # and update the text
            (x, y, w, h) = cv2.boundingRect(c)
            if with_video:
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                cv2.putText(frame, str(cv2.contourArea(c)), (x, y), font, 0.8, (255, 0, 255), 2)
            M = cv2.moments(c)
            center = (int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"]))
            #cv2.circle(frame, center, 5, (0, 0, 255), -1)
            my_blobs.append(center)
        except:
            print("Bad Rect")

    # TODO: Predict and process other targets in tracker

    dead = []
    # Update blobs
    if len(my_blobs) > 0:
        tracker.track_blobs(my_blobs, [width, height, 0, 0], current_frame)
        dead = km_tracker.track_blobs(my_blobs, [width, height, 0, 0], current_frame)

        for v in tracker.virtual_blobs:
            size = 5
            if v.got_updated:
                size = 10
                #cv2.rectangle(frame, (int(v.x), int(v.y)), (int(v.x + size), int(v.y + size)), v.color, size)
                if with_video:
                    cv2.circle(frame, (int(v.x), int(v.y)), 5,  v.color, -1)

        for v in km_tracker.virtual_blobs:
            size = 5
            if v.got_updated:
                size = 10
                # cv2.rectangle(frame, (int(v.x), int(v.y)), (int(v.x + size), int(v.y + size)), v.color, size)
                if with_video:
                    cv2.circle(frame, (int(v.x), int(v.y)), 5, v.color, -1)

                cv2.waitKey(5)

    # update even if theres noone here
    else:
        tracker.track_blobs(my_blobs, [width, height, 0, 0], current_frame)
        dead = km_tracker.track_blobs(my_blobs, [width, height, 0, 0], current_frame)

    if dead is not None:
        trail_crossed_lines(dead, frame, upper_lineheight, lower_lineheight, number_of_ppl_out, number_of_ppl_in, sending)

    # todo if we want trails
    # Todo move this into the blob
    blob_trails = True
    if blob_trails:
        trails = update_trails_old(tracker, trails, color=None)
        km_trails = update_trails(km_tracker, km_trails, (255,0,255))

    # show stuff
    if with_video:

        # add lines to cross to frame
        cv2.line(frame, (0, int(frame.shape[0] *lower_lineheight) ), (frame.shape[1], int(frame.shape[0] *lower_lineheight)), (0, 255, 0), 3)
        cv2.line(frame, (0, int(frame.shape[0] *upper_lineheight) ), (frame.shape[1], int(frame.shape[0] *upper_lineheight)), (0, 255, 0), 3)

        # trails
        cv2.add(frame, trails, frame)
        cv2.add(frame, km_trails, frame)
        # contours
        cv2.drawContours(frame, contours, -1, (0, 255, 0), 1)
        # frame
        cv2.rectangle(frame, (30, 30),
                      (width - 30, height - 30), (0, 0, 0), 2)

        # counter
        font = cv2.FONT_ITALIC
        text = 'out(' + str(number_of_ppl_in) + ')'
        text_out = 'in(' + str(number_of_ppl_out) + ')'
        cv2.putText(frame, text, (50, 50), font, 1, (255, 0, 255), 2)
        cv2.putText(frame, text_out, (50, 80), font, 1, (255, 0, 255), 2)

        cv2.imshow('ROI', frame)

    if savevideo:
        out.write(frame)

    if timer_enabled:
        print("[INFO] whole mainloop took: {}s".format(
            (datetime.datetime.now() - start).total_seconds()))

    if cv2.waitKey(10) == 27:
       break

# stop the timer and display FPS information
fps.stop()
print("[INFO] elasped time: {:.2f}".format(fps.elapsed()))
print("[INFO] approx. FPS: {:.2f}".format(fps.fps()))

#do a bit of cleanup
cap.release()
cv2.destroyAllWindows()
