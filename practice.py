import cv2
import numpy as np


def basic_camera_feed():
    cap = cv2.VideoCapture(1)
    if not cap.isOpened():
        print("Error: Could not open camera")
        exit()

    while True:
        ret, frame = cap.read()

        if not ret:
            print("Can't receive frame (stream end?). Existing ...")
            break

        cv2.imshow('Camera Feed', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


def camera_with_basic_preprocessing():
    cap = cv2.VideoCapture(1)
    if not cap.isOpened():
        print("Error: Could not open camera")
        exit()

    while True:
        ret, frame = cap.read()

        if not ret:
            print("Can't receive frame (stream end?). Existing ...")
            break

        # original Frame
        original = frame.copy()

        # convert to grayscale
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # blurs and smoothes the image
        blurred = cv2.GaussianBlur(gray, (15, 15), 0)

        # detects edges of the images
        edges = cv2.Canny(gray, 50, 150)

        # convert single channel images back to 3-channel for display
        gray_3ch = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
        blurred_3ch = cv2.cvtColor(blurred, cv2.COLOR_GRAY2BGR)
        edges_3ch = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)

        # create a 2x2 gird of images
        top_row = np.hstack([original, gray_3ch])
        bottom_row = np.hstack([blurred_3ch, edges_3ch])
        combined = np.vstack([top_row, bottom_row])

        # Resize for display
        height, width = combined.shape[:2]
        combined_resized = cv2.resize(combined, (width//2, height//2))

        cv2.imshow('Computer Vision Preprocessing', combined_resized)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


def interavtive_camera_filters():
    """Interactive camera with multiple filter options"""
    cap = cv2.VideoCapture(1)

    if not cap.isOpened():
        print("Error: Could not open camera")
        return

    current_filter = 0
    filter_names = ['Original', 'Grayscale', 'Blur', 'Edges', 'HSV', 'Sepia']

    print("Controls: ")
    print("- Press 'n' for next filter")
    print("- Press 'p' for previous filter")
    print("- Press 'q' to quit")
    print("- Press 's' to save current frame")

    while True:
        ret, frame = cap.read()

        if not ret:
            break

        if current_filter == 0:  # original
            processed = frame.copy()
        elif current_filter == 1:  # Grayscale
            processed = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            processed = cv2.cvtColor(processed, cv2.COLOR_GRAY2BGR)
        elif current_filter == 2:  # Blur
            processed = cv2.GaussianBlur(frame, (21, 21), 0)
        elif current_filter == 3:  # Edges
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            edges = cv2.Canny(gray, 50, 150)
            processed = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)
        elif current_filter == 4:    # HSV
            processed = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        elif current_filter == 5:  # sepia
            kernal = np.array([[0.272, 0.534, 0.131],
                              [0.349, 0.686, 0.168],
                              [0.393, 0.769, 0.189]])
            processed = cv2.transform(frame, kernal)

        cv2.putText(processed, f'Filter: {filter_names[current_filter]}',
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        cv2.imshow('Interative camera Filters', processed)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('n'):  # next filter
            current_filter = (current_filter + 1) % len(filter_names)
        elif key == ord('p'):  # previous filter
            current_filter = (current_filter - 1) % len(filter_names)
        elif key == ord('s'):
            filename = f'captured_frame_{filter_names[current_filter]}.jpg'
            cv2.imwrite(filename, processed)
            print(f'Saved {filename}')

    cap.release()
    cv2.destroyAllWindows()


def motion_detection_camera():
    """Simple motion detection using background subtraction"""
    cap = cv2.VideoCapture(1)

    if not cap.isOpened():
        print("Error: Camera is not opened")
        return

    ret, background = cap.read()
    if not ret:
        print("Could not read initial frame")
        return

    background = cv2.cvtColor(background, cv2.COLOR_BGR2GRAY)
    background = cv2.GaussianBlur(background, (21, 21), 0)

    print("Motion Detection Active")
    print("Press 'r' to reset background")
    print("Press 'q' to quit")

    while True:
        ret, frame = cap.read()

        if not ret:
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (21, 21), 0)

        diff = cv2.absdiff(background, gray)

        thresh = cv2.threshold(diff, 25, 255, cv2.THRESH_BINARY)[1]

        thresh = cv2.dilate(thresh, None, iterations=2)

        contours, _ = cv2.findContours(
            thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        for contour in contours:
            if cv2.contourArea(contour) < 50000:
                continue

            (x, y, w, h) = cv2.boundingRect(contour)
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
            cv2.putText(frame, 'Motion Detected', (x, y-10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        cv2.imshow('Motion Detection', frame)
        cv2.imshow('Threshold', thresh)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('r'):
            background = gray.copy()
            print("Background reset")

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    # basic_camera_feed()
    # camera_with_basic_preprocessing()
    # interavtive_camera_filters()
    motion_detection_camera()
