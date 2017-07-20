import numpy as np
import cv2
import imutils
import time
import datetime
import argparse
import device_config as CONFIG
import paho.mqtt.client as mqtt

def on_message(client, userdata, message):
    client.publish("1dsp192n-2x5i17u8-1kys/tot-in", 0)
    client.publish("1dsp192n-2x5i17u8-1kys/tot-out", 0)

try:
    client = mqtt.Client(client_id="1dsp192n-2x5i17u8-1kys")
    client.connect("mqtt.mobileresponse.se", 1883, 60)
    client.loop_start()
    client.publish("1dsp192n-2x5i17u8-1kys/tot-in", 0)
    client.publish("1dsp192n-2x5i17u8-1kys/tot-out", 0)
    client.subscribe("1dsp192n-2x5i17u8-1kys/reset", qos=0)
    client.on_message = on_message
    frame = cv2.imread("img_11.jpg")
    frame = imutils.resize(frame, width=400)
    print("a")
    ret, buf = cv2.imencode('.jpg', frame)
    print("b")
    client.publish("1dsp192n-2x5i17u8-1kys/image", frame, 0)
    print("c")
except:
    print("failed connect mqtt")

cv2.destroyAllWindows()