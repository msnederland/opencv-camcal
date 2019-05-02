import cv2
import numpy as np
import time
from camerautils import detect
from camerautils import calibrate

def nothing(x):
    # any operation
    pass

cap = cv2.VideoCapture(0)

# font = cv2.FONT_HERSHEY_COMPLEX

while True:
    _, frame = cap.read()
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    # blur = hsv
    # blur = cv2.GaussianBlur(hsv,(21,21),0)
    blur = cv2.medianBlur(hsv, 3)
    # blur = cv2.bilateralFilter(hsv, 9, 350, 350)

    # lower_red = np.array([0,187,101])
    # upper_red = np.array([119,250,179])
    lower_red = np.array([0,193,125])
    upper_red = np.array([3,255,255])

    lower_blue = np.array([91,166,62])
    upper_blue = np.array([136,255,124])

    lower_green = np.array([56,43,59])
    upper_green = np.array([87,159,137])


    mask_red = cv2.inRange(blur, lower_red, upper_red)
    mask_blue = cv2.inRange(blur, lower_blue, upper_blue)
    mask_green = cv2.inRange(blur, lower_green, upper_green)

    contours_red, _ = cv2.findContours(mask_red,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
    contours_blue, _ = cv2.findContours(mask_blue,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
    contours_green, _ = cv2.findContours(mask_green,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)

    for cnt in contours_red:

        area = cv2.contourArea(cnt)

        if(area > 1000):
            print("found one!")

            x, y, w, h = cv2.boundingRect(cnt)
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 0, 255), 4)
            rect = cv2.minAreaRect(cnt)
            box = cv2.boxPoints(rect)
            box = np.int0(box)
            # cv2.drawContours(frame, [box], 0, (0, 0, 255))

    for cnt in contours_green:

        area = cv2.contourArea(cnt)

        if(area > 1000):
            print("found one!")

            x, y, w, h = cv2.boundingRect(cnt)
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 3)
            rect = cv2.minAreaRect(cnt)
            box = cv2.boxPoints(rect)
            box = np.int0(box)
            # cv2.drawContours(frame, [box], 0, (0, 0, 255))

    for cnt in contours_blue:

        area = cv2.contourArea(cnt)

        if(area > 1000):
            print("found one!")

            x, y, w, h = cv2.boundingRect(cnt)
            cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 3)
            rect = cv2.minAreaRect(cnt)
            box = cv2.boxPoints(rect)
            box = np.int0(box)
            # cv2.drawContours(frame, [box], 0, (255, 0, 0))

    cv2.imshow("contours", frame)

    key = cv2.waitKey(1)
    if key == 27:
        break

    # cv2.namedWindow('Frame',cv2.WINDOW_NORMAL)
    # cv2.namedWindow('Mask',cv2.WINDOW_NORMAL)

    # mask_small = cv2.resize(mask,(1600,1200))
    # frame_small = cv2.resize(frame,(1600,1200))

    # cv2.imshow("Frame", frame_small)
    # cv2.imshow("Mask", mask_small)

    # key = cv2.waitKey(1)
    # if key == 27:
    #     break

cap.release()
cv2.destroyAllWindows()