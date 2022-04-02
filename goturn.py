import cv2


"""
To use GOTURN tracker you need
goturn.prototxt and goturn.caffemodel
from this repo to the working folder
https://github.com/Mogball/goturn-files
"""

if __name__ == '__main__':
    # tracker_types = ['BOOSTING', 'MIL', 'KCF', 'TLD', 'MEDIANFLOW', 'GOTURN', 'MOOSE', 'CSRT']
    tracker = cv2.TrackerGOTURN_create()

    cap = cv2.VideoCapture(0)
    res, frame = cap.read()
    bbox = cv2.selectROI(frame, False)
    res = tracker.init(boundingBox=bbox, image=frame)

    while cap.isOpened():
        res, frame = cap.read()
        if not res:
            break

        timer = cv2.getTickCount()

        res, bbox = tracker.update(frame)

        fps = cv2.getTickFrequency() / (cv2.getTickCount() - timer)

        if res:
            p1 = (int(bbox[0]), int(bbox[1]))
            p2 = (int(bbox[0] + bbox[2]), int(bbox[1] + bbox[3]))
            cv2.rectangle(frame, p1, p2, (255, 0, 0), 2, 1)
        else:
            cv2.putText(frame, 'Tracking failure detected', (100, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 255), 2)

        # cv2.putText(frame, 'GOTURN Tracker', (100, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (50, 170, 50), 2)
        # cv2.putText(frame, 'FPS: ' + str(int(fps)), (100, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (50, 170, 50), 2)
        cv2.imshow('Tracking', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
