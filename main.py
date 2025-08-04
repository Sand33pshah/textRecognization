import cv2
from matplotlib import pyplot as plt
import numpy as np
import imutils
import easyocr


img = cv2.imread('NumberPlates/mahendra.jpg')
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
plt.imshow(cv2.cvtColor(gray, cv2.COLOR_BGR2RGB))
# plt.title("GrayScale Image")
# plt.show()


bfilter = cv2.bilateralFilter(gray, 11, 17, 17)
edge = cv2.Canny(bfilter, 30, 200)
plt.imshow(cv2.cvtColor(edge, cv2.COLOR_BGR2RGB))
# plt.show()


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
# print(location)


mask = np.zeros(gray.shape, np.uint8)
newImage = cv2.drawContours(mask, [location], 0, 255, -1)
newImage = cv2.bitwise_and(img, img, mask=mask)

plt.imshow(cv2.cvtColor(newImage, cv2.COLOR_BGR2RGB))
# plt.show()


(x, y) = np.where(mask == 255)
(x1, y1) = (np.min(x), np.min(y))
(x2, y2) = (np.max(x), np.max(y))

cropped_image = gray[x1: x2+1, y1:y2+1]

plt.imshow(cv2.cvtColor(cropped_image, cv2.COLOR_BGR2RGB))
# plt.show()


reader = easyocr.Reader(['en'], gpu=False)
result = reader.readtext(cropped_image)
print(result)  # gives the number plate number


text = result[0][-2]
font = cv2.FONT_HERSHEY_SIMPLEX
res = cv2.putText(img, text=text, org=(approx[0][0][0], approx[1][0][1]+60),
                  fontFace=font, fontScale=1, color=(0, 255, 0), thickness=2, lineType=cv2.LINE_AA)
res = cv2.rectangle(img, tuple(approx[0][0]), tuple(
    approx[2][0]), (0, 255, 0), 3)

plt.imshow(cv2.cvtColor(res, cv2.COLOR_BGR2RGB))
plt.show()
