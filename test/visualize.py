import cv2
import numpy as np
import json

with open("image.json", "r", encoding="utf8") as f:
    data = json.load(f)

data = np.asarray(bytearray(data['image']), dtype="uint8")
img_data = cv2.imdecode(data, -1)
cv2.imshow("test data", img_data)
cv2.waitKey(0)
