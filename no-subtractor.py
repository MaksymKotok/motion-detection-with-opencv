import cv2


MIN_AREA = 1000
MAX_AREA = 800000

cap = cv2.VideoCapture(0)

cap.set(3, 640)
cap.set(4, 480)

frameWidth = int(cap.get(3))
frameHeight = int(cap.get(4))
_, bg = cap.read()
bg = cv2.cvtColor(bg, cv2.COLOR_BGR2GRAY)
bg = cv2.GaussianBlur(bg, (21, 21), 0)


while cap.isOpened():
    status, frame = cap.read()

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (21, 21), 0)
    diff = cv2.absdiff(bg, gray)

    thresh = cv2.threshold(diff, 30, 255, cv2.THRESH_BINARY)[1]
    thresh = cv2.dilate(thresh, None, iterations=2)

    cnts, _ = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    for contour in cnts:

        if cv2.contourArea(contour) < MIN_AREA or cv2.contourArea(contour) > MAX_AREA:
            continue

        (x, y, w, h) = cv2.boundingRect(contour)

        if w >= frameWidth and h >= frameHeight:
            continue

        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 3)

    cv2.imshow("Contours", frame)
    cv2.imshow("Threshold", thresh)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
