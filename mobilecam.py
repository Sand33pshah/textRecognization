import cv2
import numpy as np
import pytesseract
import re
import requests
from threading import Thread
import time


class PhoneCameraDetector:
    def __init__(self, phone_ip="https://192.168.1.12:8080"):
        self.phone_ip = phone_ip
        self.phone_url = f"http://{phone_ip}/video"
        self.cap = None
        self.running = False

    def connect_to_phone(self):
        """Connect to phone camera via IP Webcam app"""
        try:
            # Test connection first
            test_url = f"http://{self.phone_ip}/shot.jpg"
            response = requests.get(test_url, timeout=5)
            if response.status_code == 200:
                print(f"✅ Connected to phone camera at {self.phone_ip}")
                self.cap = cv2.VideoCapture(self.phone_url)
                return True
            else:
                print(f"❌ Failed to connect to {self.phone_ip}")
                return False
        except Exception as e:
            print(f"❌ Connection error: {e}")
            return False

    def preprocess_image(self, img):
        """Preprocess image for better license plate detection"""
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        filtered = cv2.bilateralFilter(gray, 11, 17, 17)
        edges = cv2.Canny(filtered, 30, 200)
        return gray, edges

    def detect_license_plate_contours(self, img):
        """Detect license plates using contour detection"""
        gray, edges = self.preprocess_image(img)

        contours, _ = cv2.findContours(
            edges, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        contours = sorted(contours, key=cv2.contourArea, reverse=True)[:10]

        license_plates = []

        for contour in contours:
            epsilon = 0.02 * cv2.arcLength(contour, True)
            approx = cv2.approxPolyDP(contour, epsilon, True)

            if len(approx) == 4:
                x, y, w, h = cv2.boundingRect(approx)
                aspect_ratio = w / h
                if 2.0 < aspect_ratio < 5.0 and w > 100 and h > 30:
                    license_plates.append((x, y, w, h))

        return license_plates

    def extract_text_from_plate(self, plate_img):
        """Extract text from license plate using OCR"""
        gray = cv2.cvtColor(plate_img, cv2.COLOR_BGR2GRAY)
        _, thresh = cv2.threshold(
            gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        thresh = cv2.resize(thresh, None, fx=2, fy=2,
                            interpolation=cv2.INTER_CUBIC)

        kernel = np.ones((1, 1), np.uint8)
        thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
        thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)

        config = '--oem 3 --psm 8 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789'

        try:
            text = pytesseract.image_to_string(thresh, config=config)
            text = re.sub(r'[^A-Z0-9]', '', text.upper())
            return text if len(text) > 4 else ""
        except:
            return ""

    def is_valid_license_plate(self, text):
        """Basic validation for license plate format"""
        if len(text) < 4 or len(text) > 10:
            return False

        has_letter = any(c.isalpha() for c in text)
        has_number = any(c.isdigit() for c in text)

        return has_letter and has_number

    def detect_and_read_plates(self, frame):
        """Main function to detect and read license plates"""
        plates_contour = self.detect_license_plate_contours(frame)
        detected_plates = []

        for (x, y, w, h) in plates_contour:
            plate_img = frame[y:y+h, x:x+w]

            if plate_img.size > 0:
                plate_text = self.extract_text_from_plate(plate_img)

                if plate_text and self.is_valid_license_plate(plate_text):
                    detected_plates.append({
                        'text': plate_text,
                        'bbox': (x, y, w, h),
                        'confidence': len(plate_text)
                    })

        return detected_plates

    def start_detection(self):
        """Start the detection loop"""
        if not self.connect_to_phone():
            print("❌ Failed to connect to phone camera")
            return

        print("🚀 Phone Camera License Plate Detection Started!")
        print("📱 Make sure IP Webcam app is running on your phone")
        print("Press 'q' to quit")

        while True:
            if self.cap is None:
                break

            ret, frame = self.cap.read()

            if not ret:
                print("❌ Failed to get frame from phone camera")
                break

            # Detect license plates
            plates = self.detect_and_read_plates(frame)

            # Draw results
            for plate in plates:
                x, y, w, h = plate['bbox']
                text = plate['text']

                # Draw rectangle
                cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 3)

                # Draw text with background
                text_size = cv2.getTextSize(
                    text, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)[0]
                cv2.rectangle(frame, (x, y-35),
                              (x + text_size[0] + 10, y), (0, 255, 0), -1)
                cv2.putText(frame, text, (x + 5, y - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2)

                print(f"📱 Detected: {text}")

            # Show status
            status_text = f"Phone Camera - {len(plates)} plates detected"
            cv2.putText(frame, status_text, (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

            # Display frame
            cv2.imshow('Phone Camera License Plate Detection', frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        self.cleanup()

    def cleanup(self):
        """Clean up resources"""
        if self.cap:
            self.cap.release()
        cv2.destroyAllWindows()
        print("🔴 Detection stopped")

# Alternative: USB connection method


class USBPhoneCamera:
    def __init__(self):
        self.cap = None

    def find_phone_camera(self):
        """Try to find phone camera via USB"""
        print("🔍 Searching for phone camera via USB...")

        # Try different camera indices
        for i in range(5):
            cap = cv2.VideoCapture(i)
            if cap.isOpened():
                ret, frame = cap.read()
                if ret:
                    print(f"📱 Found camera at index {i}")
                    return cap
                cap.release()

        print("❌ No phone camera found via USB")
        return None

    def start_detection(self):
        """Start detection with USB connected phone"""
        self.cap = self.find_phone_camera()

        if not self.cap:
            print("❌ Could not connect to phone camera via USB")
            return

        # Use the same detection logic as the original code
        detector = PhoneCameraDetector()

        print("🚀 USB Phone Camera Detection Started!")
        print("Press 'q' to quit")

        while True:
            ret, frame = self.cap.read()

            if not ret:
                print("❌ Failed to get frame")
                break

            # Detect license plates
            plates = detector.detect_and_read_plates(frame)

            # Draw results
            for plate in plates:
                x, y, w, h = plate['bbox']
                text = plate['text']

                cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 3)

                text_size = cv2.getTextSize(
                    text, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)[0]
                cv2.rectangle(frame, (x, y-35),
                              (x + text_size[0] + 10, y), (0, 255, 0), -1)
                cv2.putText(frame, text, (x + 5, y - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2)

                print(f"📱 Detected: {text}")

            status_text = f"USB Phone Camera - {len(plates)} plates detected"
            cv2.putText(frame, status_text, (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

            cv2.imshow('USB Phone Camera License Plate Detection', frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        self.cap.release()
        cv2.destroyAllWindows()


def main():
    print("📱 Phone Camera License Plate Detector")
    print("Choose connection method:")
    print("1. IP Webcam (WiFi)")
    print("2. USB Connection")
    print("3. Manual IP Entry")

    choice = input("Enter choice (1-3): ")

    if choice == "1":
        # Default IP for IP Webcam app
        detector = PhoneCameraDetector("192.168.1.100:8080")
        detector.start_detection()

    elif choice == "2":
        detector = USBPhoneCamera()
        detector.start_detection()

    elif choice == "3":
        phone_ip = input("Enter phone IP (e.g., 192.168.1.100:8080): ")
        detector = PhoneCameraDetector(phone_ip)
        detector.start_detection()

    else:
        print("Invalid choice")


if __name__ == "__main__":
    # Required packages:
    # pip install opencv-python pytesseract numpy requests
    main()
