import cv2
from matplotlib import pyplot as plt
import numpy as np
import imutils
import easyocr

cap = cv2.VideoCapture(1)

if not cap.isOpened():
    print("Error: Camera is not Opened")

reader = easyocr.Reader(['en'], gpu=False)

while True:
    ret, frame = cap.read()

    if not ret:
        print("Cannot read the frame (Streaming Ended ?)")
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    bfilter = cv2.bilateralFilter(gray, 11, 17, 17)
    edge = cv2.Canny(bfilter, 30, 200)

    keypoints = cv2.findContours(
        edge.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    contours = imutils.grab_contours(keypoints)
    contours = sorted(contours, key=cv2.contourArea, reverse=True)[:10]

    location = None
    for contour in contours:
        approx = cv2.approxPolyDP(contour, 10, True)
        if len(approx) == 4:
            location = approx
            break

    if location is not None:
        mask = np.zeros(gray.shape, np.uint8)
        newImage = cv2.drawContours(mask, [location], 0, 255, -1)
        newImage = cv2.bitwise_and(frame, frame, mask=mask)

        (x, y) = np.where(mask == 255)
        (x1, y1) = (np.min(x), np.min(y))
        (x2, y2) = (np.max(x), np.max(y))

        cropped_image = gray[x1: x2 + 1, y1: y2 + 1]

        result = reader.readtext(cropped_image)

        if result:
            text = result[0][-2]
            font = cv2.FONT_HERSHEY_SIMPLEX
            cv2.putText(frame, text=text, org=(approx[0][0][0], approx[1][0][1] + 60),
                        fontFace=font, fontScale=1, color=(0, 255, 0), thickness=2, lineType=cv2.LINE_AA)
            cv2.rectangle(frame, tuple(approx[0][0]),
                          tuple(approx[2][0]), (0, 255, 0), 3)

            print("Detected Text:", text)
        else:
            print("Text could not be read.")
    else:
        print("No license plate-like contour found.")

    cv2.imshow("Live Licence Detection", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
