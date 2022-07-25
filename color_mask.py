import cv2
from math import fabs
import sys
import os
import time
import numpy as np


cap = cv2.VideoCapture("outpy_30.avi")

cv2.namedWindow("track")
cv2.createTrackbar("H", "track", 0, 180, lambda a: None)
cv2.createTrackbar("S", "track", 0, 255, lambda a: None)
cv2.createTrackbar("V", "track", 0, 255, lambda a: None)

cv2.createTrackbar("HL", "track", 0, 180, lambda a: None)
cv2.createTrackbar("SL", "track", 0, 255, lambda a: None)
cv2.createTrackbar("VL", "track", 0, 255, lambda a: None)

kernel = np.ones((5, 5), np.uint8)


while cap.isOpened():
    status, frame = cap.read()

    if not status:
        cap.release()
        cap = cv2.VideoCapture("outpy_30.avi")
        status, frame = cap.read()

    # frame = cv2.bilateralFilter(frame, 9, 75, 75)
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    h = cv2.getTrackbarPos("H", "track")
    s = cv2.getTrackbarPos("S", "track")
    v = cv2.getTrackbarPos("V", "track")

    hl = cv2.getTrackbarPos("HL", "track")
    sl = cv2.getTrackbarPos("SL", "track")
    vl = cv2.getTrackbarPos("VL", "track")

    lower = np.array([hl, sl, vl])
    upper = np.array([h, s, v])

    mask = cv2.inRange(hsv, lower,  upper)
    res = cv2.bitwise_and(frame, frame, mask=mask)

    opening = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    closing = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

    print([[hl, sl, vl], [h, s, v]])

    cnts, h = cv2.findContours(closing.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

    MIN_AREA = 10000

    for contour in cnts:
        if cv2.contourArea(contour) < MIN_AREA:
            continue

        (x, y, width, height) = cv2.boundingRect(contour)
        cv2.rectangle(frame, (x, y), (x + width, y + height), (0, 255, 0), 3)

    cv2.imshow("Contours", frame)
    # cv2.imshow("Mask", mask)
    cv2.imshow("Closing", closing)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
