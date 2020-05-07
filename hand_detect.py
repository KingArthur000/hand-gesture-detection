import cv2
import numpy as np

cap = cv2.VideoCapture(0)

frame_1 = np.zeros((480,640))
lower_hsv = np.array([0,41,103])
upper_hsv = np.array([255,136,207])

lower_rgb = np.array([36,37,81])
upper_rgb = np.array([147,137,143])

while True:
    ret,frame = cap.read()
    #color conversions
    gray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
    hsv = cv2.cvtColor(frame,cv2.COLOR_BGR2HSV)
    #masks
    mask_1 = cv2.inRange(hsv,lower_hsv,upper_hsv)
    mask_2 = cv2.inRange(frame,lower_rgb,upper_rgb)

    #defining kernels
    kernel_square = np.ones((10,10),np.uint8)
    kernel_ellipse = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(5,5))

    #filters, thresholdings
    dilation = cv2.dilate(mask_1,kernel_ellipse,iterations = 1)
    erosion = cv2.erode(dilation,kernel_square,iterations = 1)    
    dilation2 = cv2.dilate(erosion,kernel_ellipse,iterations = 1)    
    filtered = cv2.medianBlur(dilation2,5)
    kernel_ellipse= cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(8,8))
    dilation2 = cv2.dilate(filtered,kernel_ellipse,iterations = 1)
    kernel_ellipse= cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(5,5))
    median = cv2.medianBlur(dilation2,5)
    ret,thresh = cv2.threshold(median,127,255,0)
    #contours
    cnts,_ = cv2.findContours(thresh,cv2.RETR_LIST,cv2.CHAIN_APPROX_SIMPLE)

    max_area=100
    ci=0	
    if len(cnts) > 0:
        for i in range(len(cnts)):
            cnt=cnts[i]
            area = cv2.contourArea(cnt)
            if(area>max_area):
                max_area=area
                ci=i  
        #getting the largest contour
        cnt = cnts[ci]
        #forming the hull
        hull = cv2.convexHull(cnt)

        cv2.drawContours(frame,[hull],-1,[0,255,0],2)

    #cv2.imshow("mask",mask_1)
    #cv2.imshow("thresh",thresh)
    #cv2.imshow("hsv",hsv)
    cv2.imshow("win",frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()