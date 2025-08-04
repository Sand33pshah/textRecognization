import cv2
import numpy as np
import pytesseract
import re

# Configure pytesseract path (uncomment and modify if needed)
# pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'


class LicensePlateDetector:
    def __init__(self):
        # Load Haar cascade for license plate detection
        # You can download this from: https://github.com/opencv/opencv/tree/master/data/haarcascades
        try:
            self.plate_cascade = cv2.CascadeClassifier(
                cv2.data.haarcascades + 'haarcascade_russian_plate_number.xml')
        except:
            # Fallback: we'll use contour detection instead
            self.plate_cascade = None
            print("Haar cascade not found, using contour detection method")

    def preprocess_image(self, img):
        """Preprocess image for better license plate detection"""
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        # Apply bilateral filter to reduce noise while preserving edges
        filtered = cv2.bilateralFilter(gray, 11, 17, 17)
        # Apply edge detection
        edges = cv2.Canny(filtered, 30, 200)
        return gray, edges

    def detect_license_plate_contours(self, img):
        """Detect license plates using contour detection"""
        gray, edges = self.preprocess_image(img)

        # Find contours
        contours, _ = cv2.findContours(
            edges, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        contours = sorted(contours, key=cv2.contourArea, reverse=True)[:10]

        license_plates = []

        for contour in contours:
            # Approximate contour to polygon
            epsilon = 0.02 * cv2.arcLength(contour, True)
            approx = cv2.approxPolyDP(contour, epsilon, True)

            # License plates are typically rectangular (4 corners)
            if len(approx) == 4:
                x, y, w, h = cv2.boundingRect(approx)

                # Check aspect ratio (license plates are wider than tall)
                aspect_ratio = w / h
                if 2.0 < aspect_ratio < 5.0 and w > 100 and h > 30:
                    license_plates.append((x, y, w, h))

        return license_plates

    def detect_license_plate_cascade(self, img):
        """Detect license plates using Haar cascade"""
        if self.plate_cascade is None:
            return []

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        plates = self.plate_cascade.detectMultiScale(gray, 1.1, 4)
        return [(x, y, w, h) for (x, y, w, h) in plates]

    def extract_text_from_plate(self, plate_img):
        """Extract text from license plate using OCR"""
        # Preprocess the plate image
        gray = cv2.cvtColor(plate_img, cv2.COLOR_BGR2GRAY)

        # Apply threshold to get better OCR results
        _, thresh = cv2.threshold(
            gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

        # Resize image for better OCR
        thresh = cv2.resize(thresh, None, fx=2, fy=2,
                            interpolation=cv2.INTER_CUBIC)

        # Apply morphological operations to clean up
        kernel = np.ones((1, 1), np.uint8)
        thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
        thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)

        # OCR configuration for license plates
        config = '--oem 3 --psm 8 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789'

        try:
            text = pytesseract.image_to_string(thresh, config=config)
            # Clean up the text
            text = re.sub(r'[^A-Z0-9]', '', text.upper())
            return text if len(text) > 4 else ""
        except:
            return ""

    def is_valid_license_plate(self, text):
        """Basic validation for license plate format"""
        if len(text) < 4 or len(text) > 10:
            return False

        # Check if it contains both letters and numbers
        has_letter = any(c.isalpha() for c in text)
        has_number = any(c.isdigit() for c in text)

        return has_letter and has_number

    def detect_and_read_plates(self, frame):
        """Main function to detect and read license plates"""
        # Try both detection methods
        plates_cascade = self.detect_license_plate_cascade(frame)
        plates_contour = self.detect_license_plate_contours(frame)

        # Combine results and remove duplicates
        all_plates = plates_cascade + plates_contour
        detected_plates = []

        for (x, y, w, h) in all_plates:
            # Extract the license plate region
            plate_img = frame[y:y+h, x:x+w]

            if plate_img.size > 0:
                # Extract text from the plate
                plate_text = self.extract_text_from_plate(plate_img)

                if plate_text and self.is_valid_license_plate(plate_text):
                    detected_plates.append({
                        'text': plate_text,
                        'bbox': (x, y, w, h),
                        # Simple confidence based on text length
                        'confidence': len(plate_text)
                    })

        return detected_plates


def main():
    # Initialize camera
    cap = cv2.VideoCapture(1)
    if not cap.isOpened():
        print("Error: Could not open camera")
        exit()

    # Initialize license plate detector
    detector = LicensePlateDetector()

    print("Real-time License Plate Detection Started!")
    print("Press 'q' to quit")

    while True:
        ret, frame = cap.read()

        if not ret:
            print("Can't receive frame (stream end?). Exiting...")
            break

        # Detect license plates in real-time
        plates = detector.detect_and_read_plates(frame)

        # Draw bounding boxes and text on detected plates
        for plate in plates:
            x, y, w, h = plate['bbox']
            text = plate['text']

            # Draw rectangle around license plate
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 3)

            # Create a background for better text visibility
            text_size = cv2.getTextSize(
                text, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)[0]
            cv2.rectangle(frame, (x, y-35),
                          (x + text_size[0] + 10, y), (0, 255, 0), -1)

            # Draw the license plate text
            cv2.putText(frame, text, (x + 5, y - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2)

            # Optional: Print to console for debugging
            print(f"Real-time Detection: {text}")

        # Show detection status on screen
        status_text = f"Detecting... ({len(plates)} plates found)"
        cv2.putText(frame, status_text, (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        # Display frame
        cv2.imshow('Real-time License Plate Detection', frame)

        # Press 'q' to quit
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Cleanup
    cap.release()
    cv2.destroyAllWindows()
    print("Detection stopped.")


if __name__ == "__main__":
    # Note: You need to install these packages:
    # pip install opencv-python pytesseract numpy
    # Also install Tesseract OCR: https://github.com/tesseract-ocr/tesseract

    main()
