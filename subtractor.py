import cv2
import sys
import os
from math import fabs
import time

MIN_AREA = 10000

cap = cv2.VideoCapture(0)

cap.set(3, 640)
cap.set(4, 480)

fW = int(cap.get(3))
fH = int(cap.get(4))

background_subtractor = cv2.createBackgroundSubtractorMOG2(history=200, varThreshold=100, detectShadows=False)


while cap.isOpened():
    status, frame = cap.read()

    mask = background_subtractor.apply(frame)
    _, mask = cv2.threshold(mask, 254, 255, cv2.THRESH_BINARY)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    for cnt in contours:
        area = cv2.contourArea(cnt)

        if area > MIN_AREA:
            x, y, w, h = cv2.boundingRect(cnt)

            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 3)
            cv2.drawContours(frame, [cnt], -1, (255, 0, 0), 3)

    cv2.imshow("Result", frame)
    cv2.imshow("Mask", mask)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
