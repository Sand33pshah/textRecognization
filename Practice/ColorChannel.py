import cv2
import numpy as np

cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Error: Camera not Opened")
    exit()

while True:
    ret, frame = cap.read()

    if not ret:
        print('cannot receive the frame (Stopped Streaming?)')
        break
    # frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    h, s, v = cv2.split(hsv)

    b, g, r = cv2.split(frame)

    h_3ch = cv2.cvtColor(h, cv2.COLOR_GRAY2BGR)
    s_3ch = cv2.cvtColor(s, cv2.COLOR_GRAY2BGR)
    v_3ch = cv2.cvtColor(v, cv2.COLOR_GRAY2BGR)

    b_3ch = cv2.cvtColor(b, cv2.COLOR_GRAY2BGR)
    g_3ch = cv2.cvtColor(g, cv2.COLOR_GRAY2BGR)
    r_3ch = cv2.cvtColor(r, cv2.COLOR_GRAY2BGR)

    # b_3ch = cv2.merge([b, np.zeros_like(b), np.zeros_like(b)])  # Blue only
    # g_3ch = cv2.merge([np.zeros_like(g), g, np.zeros_like(g)])  # Green only
    # r_3ch = cv2.merge([np.zeros_like(r), np.zeros_like(r), r])  # Red only

    top_row = np.hstack([h_3ch, s_3ch, v_3ch])
    middle_row = np.hstack([frame, hsv, np.zeros_like(frame)])
    bottom_row = np.hstack([b_3ch, g_3ch, r_3ch])
    combined = np.vstack([top_row, middle_row, bottom_row])

    cv2.putText(combined, 'Hue', (10, 50),
                cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255), 3)
    cv2.putText(combined, 'Saturation', (frame.shape[1]+10, 50),
                cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255), 3)
    cv2.putText(combined, 'Value', (2*frame.shape[1]+10, 50),
                cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255), 3)

    cv2.putText(combined, 'Original', (50, frame.shape[0] + 50),
                cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255), 3)
    cv2.putText(combined, 'HSV', (frame.shape[1]+50, frame.shape[0] + 50),
                cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255), 3)

    cv2.putText(combined, 'Blue', (10, 2 * frame.shape[0]+50),
                cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255), 3)
    cv2.putText(combined, 'Green', (frame.shape[1]+50, 2 * frame.shape[0]+50),
                cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255), 3)
    cv2.putText(combined, 'Red', (2*frame.shape[1]+10, 2 * frame.shape[0]+50),
                cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255), 3)

    height, width = combined.shape[:2]
    combined_resized = cv2.resize(combined, (width//2, height//2))

    cv2.imshow('Camera Feed', combined_resized)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
