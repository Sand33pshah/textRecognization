import cv2
import numpy as np
import pytesseract
import re
from typing import List, Tuple, Optional

# If on Windows, uncomment and set your Tesseract path
# pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'


class IndianLicensePlateRecognizer:
    def __init__(self):
        # Configure Tesseract
        self.tesseract_config = '--oem 3 --psm 8 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789'

        # Valid Indian license plate patterns
        self.plate_patterns = [
            r'^[A-Z]{2}[0-9]{2}[A-Z]{1,2}[0-9]{4}$',  # e.g., MH12AB1234
            r'^[A-Z]{2}[0-9]{2}[A-Z]{1}[0-9]{4}$',    # e.g., MH12A1234
            r'^[A-Z]{3}[0-9]{4}$',                    # e.g., DLX1234
        ]

    def preprocess_image(self, image: np.ndarray) -> np.ndarray:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        gray = clahe.apply(gray)
        gray = cv2.bilateralFilter(gray, 9, 75, 75)
        return gray

    def detect_license_plate(self, gray_image: np.ndarray) -> List[Tuple[int, int, int, int]]:
        edges = cv2.Canny(gray_image, 50, 150)
        contours, _ = cv2.findContours(
            edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        plate_candidates = []
        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)
            aspect_ratio = w / h
            area = cv2.contourArea(contour)

            if (2.0 <= aspect_ratio <= 5.0 and area > 500 and w > 80 and h > 20):
                plate_candidates.append((x, y, w, h))

        plate_candidates.sort(key=lambda x: x[2] * x[3], reverse=True)
        return plate_candidates

    def enhance_plate_region(self, plate_region: np.ndarray) -> np.ndarray:
        height, width = plate_region.shape[:2]
        if height < 32:
            scale = 32 / height
            plate_region = cv2.resize(plate_region, (int(width * scale), 32))

        kernel = np.ones((1, 1), np.uint8)
        plate_region = cv2.morphologyEx(plate_region, cv2.MORPH_CLOSE, kernel)
        plate_region = cv2.morphologyEx(plate_region, cv2.MORPH_OPEN, kernel)
        _, plate_region = cv2.threshold(
            plate_region, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

        return plate_region

    def extract_text_from_plate(self, plate_region: np.ndarray) -> str:
        enhanced = self.enhance_plate_region(plate_region)

        texts = []
        for img in [enhanced, cv2.bitwise_not(enhanced)]:
            text = pytesseract.image_to_string(
                img, config=self.tesseract_config)
            clean = text.strip().replace(" ", "").replace("\n", "")
            texts.append(clean)

        for text in texts:
            if self.validate_license_plate(text):
                return text
        return max(texts, key=len) if texts else ""

    def validate_license_plate(self, text: str) -> bool:
        text = text.upper().strip()
        for pattern in self.plate_patterns:
            if re.match(pattern, text):
                return True
        return False

    def process_video_stream(self, camera_index: int = 0):
        cap = cv2.VideoCapture(camera_index)
        if not cap.isOpened():
            print("Error: Could not open camera")
            return

        print("Live License Plate Recognition - Press 'q' to quit")

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            resized_frame = cv2.resize(frame, (640, 480))
            gray = self.preprocess_image(resized_frame)
            plate_candidates = self.detect_license_plate(gray)

            for x, y, w, h in plate_candidates[:3]:  # Try top 3 candidates
                plate_region = gray[y:y+h, x:x+w]
                license_text = self.extract_text_from_plate(plate_region)

                if license_text and self.validate_license_plate(license_text):
                    cv2.rectangle(resized_frame, (x, y),
                                  (x + w, y + h), (0, 255, 0), 2)
                    cv2.putText(resized_frame, license_text, (x, y - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
                    break

            cv2.imshow("Live License Plate Detection", resized_frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        cap.release()
        cv2.destroyAllWindows()


# Example usage
if __name__ == "__main__":
    recognizer = IndianLicensePlateRecognizer()
    recognizer.process_video_stream(0)
