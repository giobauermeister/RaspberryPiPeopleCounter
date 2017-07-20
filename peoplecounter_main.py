# import the necessary packages
from __future__ import print_function
import numpy as np
import argparse
import datetime
import json
import imutils
import device_config as CONFIG
import cv2
import time
import pt_config2
import requests
from threading import Thread
from threading import Lock
import prototype_tracker
import Queue as queue
from imutils.video import FPS


socket_init = False
#	1dsp192n-2x5i17u8-1kys
client = 0

#mqtt_timeout = datetime.datetime.now()

def on_message(client, userdata, message):
        reset_number_of_ppl()
        client.publish("1dsp192n-2x5i17u8-1kys/tot-in", 0)
        client.publish("1dsp192n-2x5i17u8-1kys/tot-out", 0)

if CONFIG.WITH_SOCKET:
    try:
        import paho.mqtt.client as mqtt
        client = mqtt.Client(client_id="1dsp192n-2x5i17u8-1kys")
        client.connect("mqtt.mobileresponse.se", 1883, 60)
        client.loop_start()
        client.publish("1dsp192n-2x5i17u8-1kys/tot-in", 0)
        client.publish("1dsp192n-2x5i17u8-1kys/tot-out", 0)
        client.subscribe("1dsp192n-2x5i17u8-1kys/reset", qos=0)
        client.on_message = on_message
        socket_init = True
    except:
        print("failed connect mqtt")

# Version of the software
version = 0.2

def get_number_of_ppl():
    global number_of_ppl_out
    global number_of_ppl_in
    return number_of_ppl_in, number_of_ppl_out

def set_number_of_ppl(new_value_in,new_value_out):
    global number_of_ppl_out
    global number_of_ppl_in
    number_of_ppl_in += new_value_in
    number_of_ppl_out += new_value_out

def reset_number_of_ppl():
    global number_of_ppl_out
    global number_of_ppl_in
    number_of_ppl_in = 0
    number_of_ppl_out = 0

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

def get_address(arg):
    try:
        ni.ifaddresses(arg)
        return ni.ifaddresses(arg)[2][0]['addr']
    except:
        print("couldnt get "+arg+" address")
        return ""


exitFlag = 0

# for counter update
class MRCounterWorker(Thread):
    def __init__(self, q):
        Thread.__init__(self)
        self.q = q

    def run(self):
        post_to_mr_job(self.q)

# for heartbeat only
class MRHeartbeatWorker(Thread):
    def __init__(self):
        Thread.__init__(self)

    def run(self):
        post_heartbeat()

def post_to_mr_job(q):
    while not exitFlag:
        # Get the work from the queue and expand the tuple
        d_tot_in = 0
        d_tot_out = 0
        updates = 0
        tot_in, tot_out = get_number_of_ppl()
        queueLock.acquire()
        if not workQueue.empty():
            while not workQueue.empty():
                delta_in, delta_out = q.get(timeout=0.3)
                d_tot_in += delta_in
                d_tot_out += delta_out
                updates += 1
            queueLock.release()

            if updates > 0:
                # Post the work to mr
                try:
                    payload = {"metadata":
                        {
                            "type": "passage",
                            "device-name": CONFIG.DEVICE_NAME,
                            "total-in": tot_in,
                            "total-out": tot_out,
                            "d-in": d_tot_in,
                            "d-out": d_tot_out
                        },
                        "workflowid": CONFIG.WORKFLOW_ID}
                    headers = {'content-type': 'application/json', 'accept': 'application/json',
                               'api-key': CONFIG.API_KEY}
                    response = requests.post(CONFIG.POST_URL, data=json.dumps(payload), headers=headers)
                except:
                    print("exception posting to mr")

        else:
            queueLock.release()
        time.sleep(CONFIG.HTTP_POST_WAIT)


def post_heartbeat():
    while not exitFlag:
        try:
            if not windows:
                payload = {"metadata":
                    {
                        "type": "heartbeat",
                        "device-name": CONFIG.DEVICE_NAME,
                        "wlan0": get_address('wlan0'),
                        "tun0": get_address('tun0'),
                        "eth0": get_address('eth0'),
                        "v": version
                    },
                    "workflowid": CONFIG.WORKFLOW_ID}

            headers = {'content-type': 'application/json', 'accept': 'application/json', 'api-key': CONFIG.API_KEY}
            response = requests.post(CONFIG.POST_URL, data=json.dumps(payload), headers=headers)
        except:
            print("exception posting to mr")
        time.sleep(CONFIG.HEARTBEAT_TIMER)

def trail_crossed_lines(dead_blobs, frame, upper_lineheight, lower_lineheight):
    d_in = 0
    d_out = 0

    for v in dead_blobs:

        # loop through trail
        p_a = v.traces[0]
        p_b = v.traces[-1]

        upper = int(frame.shape[0] * upper_lineheight)
        lower = int(frame.shape[0] * lower_lineheight)

        crossed_in = p_a[1] > upper and p_b[1] < lower
        crossed_out = p_a[1] < upper and p_b[1] > lower

        if crossed_in:
            d_in += 1

        elif crossed_out:
            d_out += 1

    queueLock.acquire()
    # if changes
    if d_in is not 0 or d_out is not 0:
        workQueue.put((d_in, d_out))
        set_number_of_ppl(d_in, d_out)
        t_in, t_out = get_number_of_ppl()
        try:
            client.publish("1dsp192n-2x5i17u8-1kys/tot-in", t_in)
            client.publish("1dsp192n-2x5i17u8-1kys/tot-out", t_out)
        except:
            print("failed to send")
    queueLock.release()

def update_trails(arg_tracker, arg_trails, color):
    c = color
    for v in arg_tracker.virtual_blobs:

        ox = None
        oy = None

        # for i in range(len(v.pred) - 1): cv2.line(frame, v.pred[i], v.pred[i + 1], v.color)
        if len(v.traces) > 2:
            # print(len(arg_tracker.traces[v.id]))
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
ap.add_argument("-s", "--slowmotion", action='store_true', default=False,
	help="run in lower fps")
ap.add_argument("-w", "--windows", action='store_true', default=False,
	help="run in lower fps")
ap.add_argument("-sv", "--savevid", action='store_true', default=False,
	help="save video")
args = vars(ap.parse_args())

# Setup
#srcTest = 'output_tst_sthlm.avi'
srcTest = 'output_new4.avi'
#srcTest = 'output_tst_rec.avi'
#srcTest = 'output_tst_big_room.avi'
#srcTest = 'output_tst_kitchen_2.avi'
#srcTest = 'output_tst_middle.avi'
#srcTest = 'output_tst_1.avi'

log_enabled = False
analyse_from_stream = args["recorded"]
with_video= args["video"]
with_analyse = args["analyse"]
with_trackbar = args["trackbars"]
slowmotion = args["slowmotion"]
windows = args["windows"]
save_vid = args["savevid"]

fourcc = 0
if save_vid:
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    writer = None

if not windows:
    import netifaces as ni

# Mobile response
post_to_mr = True


# Settings
# move to config
# Middle room camera = 2000
min_contour = CONFIG.MIN_CONTOUR
acc_w = CONFIG.TRESHOLD

#lineheight
upper_lineheight = CONFIG.UPPER_LINE_HEIGHT
lower_lineheight = CONFIG.LOWER_LINE_HEIGHT

#dist to line
dist_line = 10

# TODO: move to config???
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
tracker = prototype_tracker.BlobTracker()
# Init variables
out = 0
vs = 0
cap = 0
number_of_ppl_out = 0
number_of_ppl_in = 0
direction = ""
current_frame = 0
frame = 0

# fps
fps = FPS().start()

# Dont know
switch = '0 : PASS \n1 : STOP'

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
    if CONFIG.CAM_ROT:
        frame = rot90(frame,CONFIG.ROT_FLAG)
    c_zero = np.float32(frame.copy())
    #vs.stream.set(0, 000.0)
    width = int(frame.shape[1])
    height = int(frame.shape[0])
    #fourcc = "mp4v"
    frames = 1

else:
    _, frame = cap.read()
    frame = imutils.resize(frame, width=240)
    c_zero = np.float32(frame.copy())
    # cap.set(0, 000.0)
    width = int(frame.shape[1])
    height = int(frame.shape[0])
    #fourcc = "mp4v"
    # mjpg h624
    frames = 1


# Init stuff for the tracker
tracker.FRAME = frame.copy()
trails = np.zeros((height, width, 3)).astype(np.uint8)
start_t = time.clock()

# start threads
queueLock = Lock()
workQueue = queue.Queue()

threads = []
thread = MRCounterWorker(workQueue)
thread.start()
hbthread = MRHeartbeatWorker()
hbthread.start()
threads.append(thread)
threads.append(hbthread)

while True:
    # with a pause here the feed can replay slower
    if slowmotion:
        cv2.waitKey(30)
    trails = np.zeros((height, width, 3)).astype(np.uint8)

    # to check the time an execution of the whole mainloop takes
    if with_analyse:
        start = datetime.datetime.now()

    # detect people in the image from stream or video
    if analyse_from_stream:
        frame = vs.read()
        #frame = imutils.resize(frame,width=240)
        frame = imutils.resize(frame,width=400)
        if CONFIG.CAM_ROT:
            frame = rot90(frame, CONFIG.ROT_FLAG)
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
        #ok_to_send = (datetime.datetime.now() - mqtt_timeout.total_seconds()) > 1
       # mqtt_timeout = datetime.datetime.now()
        if CONFIG.WITH_SOCKET and socket_init:
            try:
                client.publish("1dsp192n-2x5i17u8-1kys/detected", 1)
            except:
                print("failed to send")
        dead = tracker.track_blobs(my_blobs, [width, height, 0, 0], current_frame)

    # update even if theres noone here
    else:
        dead = tracker.track_blobs(my_blobs, [width, height, 0, 0], current_frame)

    # TODO: this improved algorithm can be used to count up depending on trails
    if dead is not None:
       trail_crossed_lines(dead, frame, upper_lineheight, lower_lineheight)

    blob_trails = True
    if blob_trails:
        trails = update_trails(tracker, trails, color=None)

    # show stuff
    if with_video:

        # add lines to cross to frame
        cv2.line(frame, (0, int(frame.shape[0] *lower_lineheight) ), (frame.shape[1], int(frame.shape[0] *lower_lineheight)), (255, 255, 255), 3)
        cv2.line(frame, (0, int(frame.shape[0] *upper_lineheight) ), (frame.shape[1], int(frame.shape[0] *upper_lineheight)), (0, 0, 0), 3)

        # trails
        cv2.add(frame, trails, frame)
        # contours
        cv2.drawContours(frame, contours, -1, (0, 255, 0), 1)
        # frame
        cv2.rectangle(frame, (30, 30),
                      (width - 30, height - 30), (0, 0, 0), 2)

        # counter
        font = cv2.FONT_ITALIC
        t_in, t_out = get_number_of_ppl()
        text = 'out(' + str(t_out) + ')'
        text_out = 'in(' + str(t_in) + ')'
        cv2.putText(frame, text, (50, 50), font, 1, (255, 0, 255), 2)
        cv2.putText(frame, text_out, (50, 80), font, 1, (255, 0, 255), 2)

        cv2.imshow('ROI', frame)

        if save_vid:
            if writer is None:
                writer = cv2.VideoWriter('render_test.avi', fourcc, 30.0, (frame.shape[1],frame.shape[0]))
            # write the flipped frame
            writer.write(frame)

    if with_analyse:
        print("[INFO] whole mainloop took: {}s".format(
            (datetime.datetime.now() - start).total_seconds()))

    if cv2.waitKey(10) == 27:
        # Notify threads it's time to exit
        exitFlag = 1
        # Wait for all threads to complete
        for t in threads:
            t.join()
        break

# stop the timer and display FPS information
fps.stop()
print("[INFO] elasped time: {:.2f}".format(fps.elapsed()))
print("[INFO] approx. FPS: {:.2f}".format(fps.fps()))
#do a bit of cleanup
if with_video:
    cap.release()
else:
    vs.stop()
cv2.destroyAllWindows()
if save_vid:
    writer.release()
