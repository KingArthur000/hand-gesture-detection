import cv2
import numpy as np

def nothing(x):
    pass

cap = cv2.VideoCapture(0)

cv2.namedWindow('image')

cv2.createTrackbar('lower_R','image',0,255,nothing)
cv2.createTrackbar('lower_G','image',0,255,nothing)
cv2.createTrackbar('lower_B','image',0,255,nothing)
cv2.createTrackbar('higher_R','image',0,255,nothing)
cv2.createTrackbar('higher_G','image',0,255,nothing)
cv2.createTrackbar('higher_B','image',0,255,nothing)

while True:
    ret,frame = cap.read()
    hsv = cv2.cvtColor(frame,cv2.COLOR_BGR2HSV)
    #gray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)

    l_r = cv2.getTrackbarPos('lower_R','image')
    l_g = cv2.getTrackbarPos('lower_G','image')
    l_b = cv2.getTrackbarPos('lower_B','image')
    h_r = cv2.getTrackbarPos('higher_R','image')
    h_g = cv2.getTrackbarPos('higher_G','image')
    h_b = cv2.getTrackbarPos('higher_B','image')

    lower_rgb = np.array([l_r,l_g,l_b])
    higher_rgb = np.array([h_r,h_g,h_b])

    mask = cv2.inRange(hsv,lower_rgb,higher_rgb)
    cv2.imshow("mask",mask)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()