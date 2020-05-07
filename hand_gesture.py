import cv2
import numpy as np
import math

def pixel_dist(pt1,pt2):
    return(((pt1[0] - pt2[0])**2 + (pt1[1] - pt2[1])**2)**(1/2))

cap = cv2.VideoCapture(0)

frame_1 = np.zeros((480,640))
lower_hsv = np.array([0,58,60])
upper_hsv = np.array([183,164,171])

lower_rgb = np.array([36,37,81])
upper_rgb = np.array([147,137,143])

min_dist = 16000

while True:
    ret,frame = cap.read()
    gray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
    hsv = cv2.cvtColor(frame,cv2.COLOR_BGR2HSV)
    mask_1 = cv2.inRange(hsv,lower_hsv,upper_hsv)
    mask_2 = cv2.inRange(frame,lower_rgb,upper_rgb)

    kernel_square = np.ones((10,10),np.uint8)
    kernel_ellipse = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(5,5))

    dilation = cv2.dilate(mask_1,kernel_ellipse,iterations = 1)
    erosion = cv2.erode(dilation,kernel_square,iterations = 1)    
    dilation2 = cv2.dilate(erosion,kernel_ellipse,iterations = 1)    
    filtered = cv2.medianBlur(dilation2,5)
    kernel_ellipse= cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(8,8))
    dilation2 = cv2.dilate(filtered,kernel_ellipse,iterations = 1)
    kernel_ellipse= cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(5,5))
    median = cv2.medianBlur(dilation2,5)
    ret,thresh = cv2.threshold(median,127,255,0)

    cnts,_ = cv2.findContours(thresh,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)

    max_area=100
    ci=0	
    for i in range(len(cnts)):
        cnt=cnts[i]
        area = cv2.contourArea(cnt)
        if(area>max_area):
            max_area=area
            ci=i  
    cnt = cnts[ci]
    epsilon = 0.1*cv2.arcLength(cnt,True)
    approx = cv2.approxPolyDP(cnt,epsilon,True)
    
    hull = cv2.convexHull(cnt)

    cv2.drawContours(frame,[hull],-1,[0,255,0],2)

    hull_defect = cv2.convexHull(cnt,returnPoints = False)
    defects = cv2.convexityDefects(cnt,hull_defect)

    #min_dist = cv2.contourArea(cnt)
    defect = []
    far_point = []
    start_point = []

    for i in range(defects.shape[0]):
        s,e,f,d = defects[i,0]
        start = tuple(cnt[s][0])
        end = tuple(cnt[e][0])
        far = tuple(cnt[f][0])
        cv2.line(frame,start,end,[255,0,0],1)
        dist = int(d)
        if dist > min_dist:
            cv2.circle(frame,far,10,[100,255,255],3)
            cv2.circle(frame,start,5,(0,0,255),5)
            start_point.append(start)
            far_point.append(far)
            defect.append(dist)
        #print(cv2.arcLength(cnt,True))
    
    #for i in range(len(far_point)):
    #    cv2.line(frame,start_point[i],far_point[i],(255,0,0))

    M = cv2.moments(cnt)
    cx = int(M['m10']/M['m00'])
    cy = int(M['m01']/M['m00'])
    cv2.circle(frame,(cx,cy),3,(0,0,255),1)

    count = len(defect)
    if count == 0:
        if cv2.arcLength(cnt,True) > 800:
            cv2.putText(frame,'1',(0,470),cv2.FONT_HERSHEY_SIMPLEX,5,(255,0,0),thickness=5)
        elif cv2.arcLength(cnt,True) < 800:
            cv2.putText(frame,'0',(0,470),cv2.FONT_HERSHEY_SIMPLEX,5,(255,0,0),thickness=5)
    elif count == 1:
        cv2.putText(frame,'2',(0,470),cv2.FONT_HERSHEY_SIMPLEX,5,(255,0,0),thickness=5)
    elif count == 2:
        cv2.putText(frame,'3',(0,470),cv2.FONT_HERSHEY_SIMPLEX,5,(255,0,0),thickness=5)
    elif count == 3:
        cv2.putText(frame,'4',(0,470),cv2.FONT_HERSHEY_SIMPLEX,5,(255,0,0),thickness=5)
    elif count == 4:
        cv2.putText(frame,'5',(0,470),cv2.FONT_HERSHEY_SIMPLEX,5,(255,0,0),thickness=5)
    else:
        cv2.putText(frame,'-1',(0,470),cv2.FONT_HERSHEY_SIMPLEX,5,(255,0,0),thickness=5)

    rect = cv2.minAreaRect(cnt)
    box = cv2.boxPoints(rect)
    box = np.int0(box)
    #print(box)
    cv2.drawContours(frame,[box],0,(0,0,255),2)

    cv2.imshow("win",frame)
    #cv2.imshow("mask",mask_1)
    #cv2.imshow("thresh",thresh)
    #cv2.imshow("hsv",hsv)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()