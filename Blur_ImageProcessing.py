import cv2
import numpy as np

cap = cv2.VideoCapture(1)
blur_level = 1
sigma_Value = 0

if not cap.isOpened():
    print("Error: Camera is not Opened")
    exit()

print("Press '+' to increase blur, '-' to decrease blur, 'q' to quit")

while True:
    ret, frame = cap.read()

    if not ret:
        print("Can't Read the Frame (Stream End?)")
        break

    # copying the original Frame fro reference
    original = frame.copy()

    # converting the frame to gray scale to simplifiy the intensity of image data and reduce volume
    # RGB has 3 Channel and gray has only 1 Channel. (only represent the intensity and brightness)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # adding blur to process the image and smoothe out the frame to create clarity.

    # Calculate kernel size (must be odd)
    kernel_size = max(1, blur_level * 2 + 1)

    blur = cv2.GaussianBlur(gray, (kernel_size, kernel_size), sigma_Value)

    # converting gray image back to 3 channel for display
    blur = cv2.cvtColor(blur, cv2.COLOR_GRAY2BGR)

    # creating vertical stack fro good visual
    combined = np.vstack([original, blur])

    height, width = combined.shape[:2]
    combined_resized = cv2.resize(combined, (width//2, height//2))

    cv2.putText(combined_resized, f'Kernel Size: {kernel_size}x{kernel_size} | Sigma Value: {sigma_Value}', (
        10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    cv2.imshow('Camera Feed', combined_resized)

    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break
    elif key == ord('+'):
        blur_level = min(blur_level+1, 25)
    elif key == ord('-'):
        blur_level = max(blur_level-1, 1)
    elif key == ord('a'):
        sigma_Value = 5
    elif key == ord('s'):
        sigma_Value = 9
    elif key == ord('d'):
        sigma_Value = 0
cap.release()
cv2.destroyAllWindows()
