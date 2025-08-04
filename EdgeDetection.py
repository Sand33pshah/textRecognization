import cv2
import numpy as np

cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Error: Camera is not Opened")
    exit()

while True:
    ret, frame = cap.read()

    if not ret:
        print("Cannot read the frame (Streaming Ended?)")
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    edges1 = cv2.Canny(gray, 30, 80)
    edges2 = cv2.Canny(gray, 50, 150)
    edges3 = cv2.Canny(gray, 100, 200)
    edges4 = cv2.Canny(gray, 150, 250)

    edges1 = cv2.cvtColor(edges1, cv2.COLOR_GRAY2BGR)
    edges2 = cv2.cvtColor(edges2, cv2.COLOR_GRAY2BGR)
    edges3 = cv2.cvtColor(edges3, cv2.COLOR_GRAY2BGR)
    edges4 = cv2.cvtColor(edges4, cv2.COLOR_GRAY2BGR)

    top_row = np.hstack([edges1, edges2])
    bottom_row = np.hstack([edges3, edges4])
    combined = np.vstack([top_row, bottom_row])

    height, width = combined.shape[:2]
    combined_resized = cv2.resize(combined, (width//2, height//2))

    cv2.imshow('Camera Feed', combined_resized)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break


cap.release()
cv2.destroyAllWindows()
