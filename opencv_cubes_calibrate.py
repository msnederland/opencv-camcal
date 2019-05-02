import cv2
import numpy as np
import time
from camerautils import detect
from camerautils import calibrate

def nothing(x):
    # any operation
    pass

cap = cv2.VideoCapture(0)

cv2.namedWindow("Trackbars")
cv2.createTrackbar("L-H", "Trackbars", 0, 180, nothing)
cv2.createTrackbar("U-H", "Trackbars", 180, 180, nothing)
cv2.createTrackbar("L-S", "Trackbars", 66, 255, nothing)
cv2.createTrackbar("U-S", "Trackbars", 255, 255, nothing)
cv2.createTrackbar("L-V", "Trackbars", 134, 255, nothing)
cv2.createTrackbar("U-V", "Trackbars", 243, 255, nothing)

# font = cv2.FONT_HERSHEY_COMPLEX

while True:
    _, frame = cap.read()
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    blur = cv2.medianBlur(hsv, 3)

    l_h = cv2.getTrackbarPos("L-H", "Trackbars")
    l_s = cv2.getTrackbarPos("L-S", "Trackbars")
    l_v = cv2.getTrackbarPos("L-V", "Trackbars")
    u_h = cv2.getTrackbarPos("U-H", "Trackbars")
    u_s = cv2.getTrackbarPos("U-S", "Trackbars")
    u_v = cv2.getTrackbarPos("U-V", "Trackbars")

    lower_red = np.array([l_h,l_s,l_v])
    upper_red = np.array([u_h,u_s,u_v])

    mask = cv2.inRange(blur, lower_red, upper_red)

    cv2.namedWindow('Frame',cv2.WINDOW_NORMAL)
    cv2.namedWindow('Mask',cv2.WINDOW_NORMAL)

    mask_small = cv2.resize(mask,(1024,768))
    frame_small = cv2.resize(frame,(1024,768))
    cv2.imshow("Frame", frame_small)
    cv2.imshow("Mask", mask_small)

    key = cv2.waitKey(1)
    if key == 27:
        break

cap.release()
cv2.destroyAllWindows()