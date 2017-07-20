DEVICE_NAME = "bosbecppl3"
# Configuration for detector
UPPER_LINE_HEIGHT = 0.32
LOWER_LINE_HEIGHT = 0.48
TRESHOLD =0.05
MIN_CONTOUR = 3000

# Mobileresponse settings
WORKFLOW_ID = "7a832bec-dbb6-4b50-89fd-a319a25f8e4f"
API_KEY = "149b618e-0777-4d65-84c7-ea184de8e90d"
POST_URL = 'https://rest.mobileresponse.io/1/workflows'

# The time between heartbeats
HEARTBEAT_TIMER = 10
#the time between http posts
HTTP_POST_WAIT = 1

# rotation
# 1 = cw
# 2 = ccw
ROT_FLAG = 2
CAM_ROT = True

# With socket gui?
WITH_SOCKET = False
SOCKET_URL = 'support-chat.mobileresponse.se'
SOCKET_PORT = 3000