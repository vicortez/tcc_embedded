# Usage:
# python motion_tracking.py [--video ../videos/example_01.mp4] [--roi 300,300,1000,720]
# import the necessary packages
import argparse
import datetime
import imutils
import time
import cv2
from collections import deque
from sys import float_info
import numpy as np


# check if two line segments have an intersection
def segment_intersection(line1, line2):
    class Point:
        def __init__(self, x, y):
            self.x = x
            self.y = y

    def ccw(A, B, C):
        return (C.y - A.y) * (B.x - A.x) > (B.y - A.y) * (C.x - A.x)

    def intersect(A, B, C, D):
        return ccw(A, C, D) != ccw(B, C, D) and ccw(A, B, C) != ccw(A, B, D)

    x1 = Point(line1[0][0], line1[0][1])
    y1 = Point(line1[1][0], line1[1][1])
    x2 = Point(line2[0][0], line2[0][1])
    y2 = Point(line2[1][0], line2[1][1])
    return intersect(x1, y1, x2, y2)


# construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-v", "--video", help="path to the video file")
ap.add_argument("-a", "--min-area", type=int, default=1000, help="minimum area size")
ap.add_argument("-r", "--roi", help="region of interest: startCol,startLin,endCol,endLin")
ap.add_argument("-b", "--buffer", type=int, default=2, help="max buffer size")
args = vars(ap.parse_args())


# if the video argument is None, then we are reading from webcam
if args.get("video", None) is None:
    print("[INFO] warming up...")
    camera = cv2.VideoCapture(0)
    time.sleep(0.25)
# otherwise, we are reading from a video file
else:
    camera = cv2.VideoCapture(args["video"])

if args.get("roi"):
    (startCol, startLin, endCol, endLin) = args["roi"].split(',')
    startCol = int(startCol)
    startLin = int(startLin)
    endCol = int(endCol)
    endLin = int(endLin)

# initialize the first frame in the video stream
firstFrame = None

# initialize the frame dimensions (we'll set them as soon as we read
# the first frame from the video)
W = None
H = None

# initialize the list of tracked points, the frame counter,
# and the coordinate deltas
pts = deque(maxlen=args["buffer"])
(dX, dY) = (0, 0)
direction = ""

# loop over the frames of the video
while True:
    # grab the current frame and initialize the occupied/unoccupied
    # text
    (grabbed, frame) = camera.read()

    # if the frame could not be grabbed, then we have reached the end
    # of the video
    if not grabbed:
        break

    # crop ROI (region of interest) from image if required
    if args.get("roi"):
        frame = frame[startLin:endLin, startCol:endCol]

    # resize the frame
    frame = imutils.resize(frame, width=500)

    # if the frame dimensions are empty, set them
    if W is None or H is None:
        (H, W) = frame.shape[:2]

    # convert the frame to grayscale, and blur it
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (21, 21), 0)

    # if the first frame is None, initialize it
    if firstFrame is None:
        print("[INFO] starting background model...")
        firstFrame = gray.copy().astype("float")
        continue

    # draw a diagonal line in the center of the frame -- once an
    # object crosses this line we will determine whether they were
    # moving 'up' or 'down'
    crossline = ((0, 0), (W, H))
    cv2.line(frame, crossline[0], crossline[1], (0, 255, 255), 2)

    # accumulate the weighted average between the current frame and
    # previous frames, then compute the difference between the current
    # frame and running average
    cv2.accumulateWeighted(src=gray, dst=firstFrame, alpha=0.5)
    frameDelta = cv2.absdiff(gray, cv2.convertScaleAbs(firstFrame))
    thresh = cv2.threshold(frameDelta, 5, 255, cv2.THRESH_BINARY)[1]

    # dilate the thresholded image to fill in holes, then find contours
    # on thresholded image
    thresh = cv2.dilate(thresh, None, iterations=2)
    (_, cnts, _) = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # only proceed if at least one contour was found
    if len(cnts) > 0:
        # find the largest contour in the mask
        c = max(cnts, key=cv2.contourArea)

        # if the contour is too small, ignore it
        if cv2.contourArea(c) >= args["min_area"]:

            # compute the bounding box for the contour, draw it on the frame,
            # and update the text
            (x, y, w, h) = cv2.boundingRect(c)
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

            # find object's centroid
            CoordXCentroid = (x + x + w) // 2
            CoordYCentroid = (y + y + h) // 2

            ObjectCentroid = (CoordXCentroid, CoordYCentroid)
            cv2.circle(frame, ObjectCentroid, 1, (0, 255, 0), 5)
            pts.appendleft(ObjectCentroid)
        else:
            pts.clear()
    else:
        pts.clear()

    # show the frame to our screen and increment the frame counter
    cv2.imshow("Frame", frame)
    key = cv2.waitKey(1) & 0xFF

    # if the 'q' key is pressed, stop the loop
    if key == ord("q"):
        break

    # track the last two points
    if len(pts) > 1:
        # compute the difference between the x and y coordinates
        objline = (pts[0], pts[1])
        # check if there is an intersection between the two lines
        if segment_intersection(crossline, objline):
            dX = pts[0][0] - pts[1][0]
            dY = pts[0][1] - pts[1][1]
            (dirX, dirY) = ("", "")

            # ensure there is significant movement in the
            # x-direction
            if np.abs(dX) > 2:
                dirX = "East" if np.sign(dX) == 1 else "West"

            # ensure there is significant movement in the
            # y-direction
            if np.abs(dY) > 2:
                dirY = "South" if np.sign(dY) == 1 else "North"

            # handle when both directions are non-empty
            if dirX != "" and dirY != "":
                direction = "{}-{}".format(dirY, dirX)

            # otherwise, only one direction is non-empty
            else:
                direction = dirX if dirX != "" else dirY

            # draw the connecting lines
            cv2.line(frame, pts[0], pts[1], (0, 0, 255), 2)

            # show the movement deltas and the direction of movement on
            # the frame
            cv2.putText(frame, direction, (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
                        0.65, (0, 0, 255), 3)
            cv2.putText(frame, "dx: {}, dy: {}".format(dX, dY),
                        (10, frame.shape[0] - 10), cv2.FONT_HERSHEY_SIMPLEX,
                        0.35, (0, 0, 255), 1)
            timestamp = datetime.datetime.now()
            cv2.imshow(timestamp.strftime('%Y-%m-%d_%H-%M-%S.jpg'), frame)

# cleanup the camera and close any open windows
camera.release()
cv2.destroyAllWindows()
