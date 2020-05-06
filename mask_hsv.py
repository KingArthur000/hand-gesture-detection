import cv2
import numpy as np

def nothing(x):
    pass

cap = cv2.VideoCapture(0)

cv2.namedWindow('image')

cv2.createTrackbar('lower_H','image',0,255,nothing)
cv2.createTrackbar('lower_S','image',0,255,nothing)
cv2.createTrackbar('lower_V','image',0,255,nothing)
cv2.createTrackbar('higher_H','image',0,255,nothing)
cv2.createTrackbar('higher_S','image',0,255,nothing)
cv2.createTrackbar('higher_V','image',0,255,nothing)

while True:
    ret,frame = cap.read()
    hsv = cv2.cvtColor(frame,cv2.COLOR_BGR2HSV)
    #gray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)

    l_h = cv2.getTrackbarPos('lower_H','image')
    l_s = cv2.getTrackbarPos('lower_S','image')
    l_v = cv2.getTrackbarPos('lower_V','image')
    h_h = cv2.getTrackbarPos('higher_H','image')
    h_s = cv2.getTrackbarPos('higher_S','image')
    h_v = cv2.getTrackbarPos('higher_V','image')

    lower_hsv = np.array([l_h,l_s,l_v])
    higher_hsv = np.array([h_h,h_s,h_v])

    mask = cv2.inRange(hsv,lower_hsv,higher_hsv)
    cv2.imshow("mask",mask)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
