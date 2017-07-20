import numpy as np
import cv2
import munkres
import pt_config2

# Configuration constants.  Change these in pt_config
BLOB_LIFE = pt_config2.BLOB_LIFE  # life of blob in frames, if not seen
EDGE_THRESHOLD = pt_config2.EDGE_THRESHOLD  # border of image, in pixels, which is regarded as out-of-frame
DISTANCE_THRESHOLD = pt_config2.DISTANCE_THRESHOLD  # distance threshold, in pixels. If blob is further than this from previous position, update is ignored
MOVE_LIMIT = pt_config2.MOVE_LIMIT  # maximum velocity of the blob. If outside this limit, velocity is disabled
MATCH_DISTANCE = pt_config2.MATCH_DISTANCE  # maximum distance between blobs in the Hungarian algorithm matching step
KALMAN_TRESHOLD = 20
KALMAN_DEAD_TRESHOLD = 30


FRAME = 0

blob_id = 0


class VirtualBlob:
    """
    Represents a single pedestrian blob.
    """

    def __init__(self, x, y):
        """
        Create a new blob at the given (x,y) co-ordinate (in pixels). Each blob has a unique
        ID number, and a random color (for visulisation)
        """
        #print("new")
        global blob_id
        self.crossed_upper = False
        self.crossed_lower = False
        self.counted = False
        self.x = x
        self.y = y
        self.dx = 0
        self.dy = 0
        self.life = KALMAN_DEAD_TRESHOLD
        self.got_updated = False
        self.color = (np.random.randint(0, 255), np.random.randint(0, 255), np.random.randint(0, 255))
        self.id = blob_id
        blob_id = blob_id + 1

        #traces
        self.traces = []

        # kalman
        self.meas = []
        self.pred = []
        self.mp = np.array((2, 1), np.float32)  # measurement
        self.tp = np.zeros((2, 1), np.float32)  # tracked / prediction
        # self.frame = np.zeros((640, 480, 3), np.uint8)  # drawing canvas

        # kalman
        self.kalman = cv2.KalmanFilter(4, 2)
        self.kalman.measurementMatrix = np.array([[1, 0, 0, 0], [0, 1, 0, 0]], np.float32)
        self.kalman.transitionMatrix = np.array([[1, 0, 1, 0], [0, 1, 0, 1], [0, 0, 1, 0], [0, 0, 0, 1]], np.float32)
        self.kalman.processNoiseCov = np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]],
                                               np.float32) * 0.03

    def update_location(self, x, y):
        """Update the current state of the blob to the new given position, if it is
        not too far away (<DISTANCE_THRESHOLD away) from the previous position"""
        if abs(x - self.x) < DISTANCE_THRESHOLD and abs(y - self.y) < DISTANCE_THRESHOLD:
            self.dx = 0.65 * self.dx + 0.35 * (x - self.x)
            self.dy = 0.65 * self.dy + 0.35 * (y - self.y)
            self.x = 0.6 * self.x + 0.4 * x
            self.y = 0.6 * self.y + 0.4 * y
            self.got_updated = True
            self.life = KALMAN_DEAD_TRESHOLD

    def update(self):
        self.life = KALMAN_DEAD_TRESHOLD
        self.got_updated = True

    def update_speed_and_direction_kalman(self):
        if (len(self.pred) > KALMAN_TRESHOLD):
            predicted_location = self.pred[-1]
            self.dx = 0.65 * self.dx + 0.35 * (predicted_location[0] - self.x)
            self.dy = 0.65 * self.dy + 0.35 * (predicted_location[1] - self.y)
        elif len(self.pred) < KALMAN_TRESHOLD and len(self.meas) > 1:
            location = self.meas[-1]
            self.dx = 0.65 * self.dx + 0.35 * (location[0] - self.x)
            self.dy = 0.65 * self.dy + 0.35 * (location[1] - self.y)

    def set_location(self, x, y):
        """Change the position of the blob _without_ any distance filtering or velocity calculation."""
        self.x = x
        self.y = y

    def move(self):
        """Apply the current estimated velocity to the blob; used when the blob is not observed in the scene"""
        if abs(self.dx) < MOVE_LIMIT and abs(self.dy) < MOVE_LIMIT:
            self.x += self.dx
            self.y += self.dy

    def kalman_move(self):
        if(len(self.pred) > KALMAN_TRESHOLD):
            predicted_location = self.pred[-1]
            self.x = predicted_location[0]
            self.y = predicted_location[1]
        elif len(self.pred) < KALMAN_TRESHOLD and len(self.meas) > 1:
            self.x = self.meas[-1][0]
            self.y = self.meas[-1][1]

    def decay(self):
        self.life = self.life - 1
        return self.life <= 0

    def __repr__(self):
        return "(%d, %d, %d, %d)" % (self.x, self.y, self.dx, self.dy)

    # kalman
    def addPoint(self, x, y):
        self.mp = np.array([[np.float32(x)], [np.float32(y)]])
        self.meas.append((x, y))

    #def paint(self):
        #for i in range(len(meas)): cv2.circle(frame, meas[i - 1], 1, (255, 255, 255), -1)
        # for i in range(len(pred)): cv2.circle(frame, pred[i-1],1, (0, 0, 200),-1)

    def reset(self):
        self.meas = []
        self.pred = []
        #self.frame = np.zeros((400, 400), np.uint8)


class BlobTracker:
    """The tracker object, which keeps track of a collection of pedestrian blobs"""

    def __init__(self):
        """Initialise a new, empty tracker"""
        self.virtual_blobs = []
        #self.traces = {}
        self.frame = 0
        self.is_inited = False

    def init_blobs(self, blobs, fnum):
        """Initialise a set of blobs, from a list of initial (x,y) co-ordinates, in the format
        [(x,y), (x,y), ... ] """
        # initialise virtual blobs to be blobs
        self.virtual_blobs  = []
        for blob in blobs:
            v = VirtualBlob(blob[0], blob[1])
            self.virtual_blobs.append(v)
            v.traces.append((v.x, v.y, fnum))
        self.is_inited = True

        # returns true is this blob is within the frame

    def check_frame(self, blob, frame):
        """Given an (x,y) co-ordinated, check if that position is inside the central frame (i.e. is
        not inside the border region"""
        # Check Frame
        in_frame = False
        # left
        if blob[0] < frame[0] + EDGE_THRESHOLD:
            in_frame = True
        # right
        if blob[0] > frame[2] - EDGE_THRESHOLD:
            in_frame = True
        # top
        if blob[1] < frame[1] + EDGE_THRESHOLD:
            in_frame = True
        # bottom
        if blob[1] > frame[3] - EDGE_THRESHOLD:
            in_frame = True

        return in_frame

    def track_blobs(self, blobs, frame, fnum):
        """Main update call. Takes a list of new, observed blob co-ordinates, a rectangular frame specifier of the form
         [left, bottom, right, top] and a frame number, and updates the positions of the virtual blobs."""


        # initialise if not already done so
        if not self.is_inited:
            self.init_blobs(blobs, fnum)
            return

        if (len(blobs) > 0):
            # get max length of blob lists
            max_size = max(len(blobs), len(self.virtual_blobs))

            distance_matrix = np.zeros((max_size, max_size))

            # move it
            for v in self.virtual_blobs:
                v.kalman_move()

            # compute distance matrix
            for i in range(max_size):
                if i >= len(blobs):
                    distance_matrix[i, :] = 0
                # no matching blob/virtual blob
                else:
                    for j in range(max_size):
                        if j >= len(self.virtual_blobs):
                            distance_matrix[i, j] = 0
                        else:
                            dx = blobs[i][0] - self.virtual_blobs[j].x
                            dy = blobs[i][1] - self.virtual_blobs[j].y
                            distance_matrix[i, j] = np.sqrt(dx ** 2 + dy ** 2)

            copy_distances = np.array(distance_matrix)

            m = munkres.Munkres()
            ot = m.compute(distance_matrix)
            rows = [t[1] for t in ot]

        # clear the update flag
        for v in self.virtual_blobs:
            v.got_updated = False

        if(len(blobs) > 0):

            # blobs on rows
            for i, matching_virtual in enumerate(rows):
                if i < len(blobs):
                    blob = blobs[i]
                    if matching_virtual < len(self.virtual_blobs):
                        if copy_distances[i][matching_virtual] < MATCH_DISTANCE:
                            vblob = self.virtual_blobs[matching_virtual]
                            #kalman
                            vblob.addPoint(blob[0], blob[1])
                            vblob.update()
                            #without
                            #self.virtual_blobs[matching_virtual].update_location(blob[0], blob[1])
                        elif self.check_frame(blob, frame):
                            v = VirtualBlob(blob[0], blob[1])
                            self.virtual_blobs.append(v)
                            v.traces.append((v.x, v.y, fnum))
                            # add to list
                    else:
                        # new baby blobs!
                        spawn = False
                        # left
                        if blob[0] < frame[0] + EDGE_THRESHOLD:
                            spawn = True
                        # right
                        if blob[0] > frame[2] - EDGE_THRESHOLD:
                            spawn = True
                        # top
                        if blob[1] < frame[1] + EDGE_THRESHOLD:
                            spawn = True
                        # bottom
                        if blob[1] > frame[3] - EDGE_THRESHOLD:
                            spawn = True

                        if spawn:
                            v = VirtualBlob(blob[0], blob[1])
                            self.virtual_blobs.append(v)
                            v.traces.append((v.x, v.y, fnum))
                        else:
                            pass

        # deal with un-updated blobs
        graveyard = []
        for v in self.virtual_blobs:
            if not v.got_updated:

                # move, and reduce life counter
                if v.decay():
                    print("Virtual blob %s finally died." % v)
                    graveyard.append(v)

            #kalman
            v.kalman.correct(v.mp)
            v.tp = v.kalman.predict()
            v.pred.append((int(v.tp[0]), int(v.tp[1])))
            v.update_speed_and_direction_kalman()

            # append trace of blob movement
            if len(v.pred) > KALMAN_TRESHOLD:
                v.traces.append((v.pred[-1][0], v.pred[-1][1], fnum))
            elif len(v.pred) < KALMAN_TRESHOLD and len(v.meas)>1:
                v.traces.append((v.pred[-1][0], v.pred[-1][1], fnum))
#                self.traces[v.id].append((v.meas[-1][0], v.meas[-1][1], fnum))


        # clean up the bodies
        for v in graveyard:
            #print("remove blob",v)
            #self.traces[v.id] = []
            self.virtual_blobs.remove(v)

        return graveyard