import cv2
import numpy as np

def nothing(x):
    pass

cap = cv2.VideoCapture(0)

cv2.namedWindow('image')

cv2.createTrackbar('lower','image',0,255,nothing)
cv2.createTrackbar('higher','image',0,255,nothing)

while True:
    ret,frame = cap.read()
    #hsv = cv2.cvtColor(frame,cv2.COLOR_BGR2HSV)
    gray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)

    l_g = cv2.getTrackbarPos('lower','image')
    h_g = cv2.getTrackbarPos('higher','image')

    lower_gray = np.array([l_g])
    higher_gray = np.array([h_g])

    mask = cv2.inRange(gray,lower_gray,higher_gray)
    cv2.imshow("mask",mask)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()