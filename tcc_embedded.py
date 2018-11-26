# -*- coding: utf-8 -*-
"""
Created on Wed Jun 20 14:57:48 2018

@author: victor cortez
"""

import numpy as np
import cv2
#import classifier
#import classifier2
import functions as vic
import time


# GLOBAL VARIABLES ============================================================

#VID_DIRECTORY = "D:\\Users\\f202897\\Desktop\\vm-master\\Videos\\"
#VID_DIRECTORY = "C:\\Users\\victor\\Desktop\\vm-master\\Videos\\"
#VID_DIRECTORY = "C:\\Users\\victo\\Desktop\\"
VID_DIRECTORY = "C:\\Users\\victor\\Desktop\\tcc\\videos\\"
#VID_DIRECTORY = ""

#kernels
KERNEL5=cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(5,5))
KERNEL7=cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(7,7))
KERNEL11=cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(11,11))
KERNEL19=cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(19,19))
KERNEL23=cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(23,23))
KERNEL27=cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(27,27))
KERNEL29=cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(29,29))

#area of interest
# points = x,y (l-r, t-b)
#roi: top right, top left, bottom left, bottom right
#ROI_CORNERS = np.array([[(900,360),(20,360), (20,720), (900,720)]], dtype=np.int32)
ROI_CORNERS = np.array([[(800,360),(0,360), (0,720), (800,720)]], dtype=np.int32)

#LINE_POINTS = [(100,600),(1000,600)]
LINE_POINTS = [(400,0),(400,720)]
CAR_FLOW_ORIENTATION='horizontal'

FASTMODE = False
DEBUG=False
DISPLAY_VIDEO=True
#==============================================================================


cap = cv2.VideoCapture(VID_DIRECTORY+'imd3.mov')
#out = cv2.VideoWriter('outpy.avi',cv2.VideoWriter_fourcc('M','J','P','G'), 30, (1280,720))
fgbg = cv2.createBackgroundSubtractorMOG2()
#detectShadows=False

#benchmarking
timeslist=[]
timelogs=[(0,0,"0")]

framecount=0

#each element contains the history of each center detected
points_history=[]
#==============================================================================
while(1):
    
    millis1 = time.time()
    
    ret, frame = cap.read()
    
    if frame is None:
        print("none")
        break
    
    framecount+=1
    framecopy=frame.copy()

    
    if(FASTMODE):
    
        #getting ROI ==========================================================
        # mask defaulting to black for 3-channel and transparent for 4-channel
        mask = np.zeros(frame.shape, dtype=np.uint8)
        # fill the ROI so it doesn't get wiped out when the mask is applied
        channel_count = frame.shape[2]  # i.e. 3 or 4 depending on your image
        ignore_mask_color = (255,)*channel_count
        cv2.fillConvexPoly(mask, ROI_CORNERS, ignore_mask_color)
        # apply the mask
        roi = cv2.bitwise_and(frame, mask)
        #======================================================================
    else:
        roi=framecopy[ROI_CORNERS[0][1][1]:ROI_CORNERS[0][2][1],ROI_CORNERS[0][1][0]:ROI_CORNERS[0][0][0]]
        framecopy=roi.copy()
        

    
    # bluring =================================================================
    mblur = cv2.medianBlur(roi,5)
    
    fgmaskmblur=fgbg.apply(mblur)
    
    ret,fgmaskmblur = cv2.threshold(fgmaskmblur,127,255,cv2.THRESH_BINARY)
    #==========================================================================
    
    # morphology manipulation  ================================================
    openingmblur = cv2.morphologyEx(fgmaskmblur, cv2.MORPH_OPEN, KERNEL7)
    
    dilationmblur = cv2.morphologyEx(openingmblur,cv2.MORPH_DILATE, KERNEL29)
    morphresult = dilationmblur
    #==========================================================================
    
    
    #filling blobs ============================================================
    
    #mblur
    h, w = morphresult.shape[:2]
    mask = np.zeros((h+2, w+2), np.uint8)
    im_floodfill=morphresult.copy()
    cv2.floodFill(im_floodfill, mask, (0,0), 255);
    # Invert floodfilled image
    im_floodfill_inv = cv2.bitwise_not(im_floodfill)
    # Combine the two images to get the foreground.
    filledimg = morphresult | im_floodfill_inv
    
    fillingresult = filledimg
    #==========================================================================
    
    # contours ================================================================
    image22, contours, hierachy = cv2.findContours(fillingresult.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    imagehull = image22.copy()
    hull = [cv2.convexHull(c) for c in contours]
    #clearing small blobs
    for index, el in enumerate(hull):
        area = cv2.contourArea(el)
        if area < 1500 or area > 50000:
            del hull[index]
        
    
    blank=np.zeros(image22.copy().shape,np.uint8)
    #converting image back to 3 channels to display border colours
    image22=cv2.cvtColor(image22,cv2.COLOR_GRAY2BGR)
    #image23=cv2.cvtColor(image23,cv2.COLOR_GRAY2BGR)
    imagehull=cv2.cvtColor(imagehull,cv2.COLOR_GRAY2BGR)
    
    # rectangles ==============================================================
    rectangles=[cv2.boundingRect(h) for h in hull]
    #==========================================================================
    
    # drawing contours ========================================================
    image22 = cv2.drawContours(image22,contours,-1,(0,255,0))
    blank=cv2.drawContours(blank,hull,-1,255,cv2.FILLED)
    #image23 = cv2.drawContours(image23,contours2,-1,(0,255,0))
    imagehull = cv2.drawContours(imagehull,hull,-1,(0,255,0),cv2.FILLED)
    contouringresult=blank
    #==========================================================================
    
    # drawing rectangles ======================================================
    [cv2.rectangle(framecopy,(rect[0],rect[1]),(rect[0]+rect[2],rect[1]+rect[3]),(0,0,255),3) for rect in rectangles]
    #==========================================================================
    
    
    # counting ================================================================
    #para cada contorno, checar se ele ja existia no frame anterior. se nao,
    #aloca-lo na lista, se sim, adicionar a um historico de pontos existente.
    
    
    # tracking resulting blobs
    for index, el in enumerate(hull):
        M=cv2.moments(el)
        cx = int(M["m10"] / M["m00"])
        cy = int(M["m01"] / M["m00"])
        p=(cx,cy)
        cv2.circle(framecopy, p, 7, (0, 255, 255), -1)
            
        if vic.point_existed(p,points_history):
            pos = vic.get_point_pos(p,points_history)
            points_history[pos].append(p)
                    
        else:
            points_history.append([p])
            
    # printing trails
    if(DEBUG and len(points_history) >= 1):
        for one_point_history in points_history:
            for point in one_point_history:
                cv2.circle(framecopy,point,7,(255,0,0),-1)
    
    # counting and classification
    point_crossed,point,point_history=vic.point_crossed_line(points_history,LINE_POINTS,CAR_FLOW_ORIENTATION)
    if point_crossed:
        cv2.line(framecopy,LINE_POINTS[0],LINE_POINTS[1],(0,255,0),5)
        vic.send_to_server(point,rectangles,roi)
        vic.drop_point(point_history,points_history)
    else:
        cv2.line(framecopy,LINE_POINTS[0],LINE_POINTS[1],(255,0,0),5)
        
    #==========================================================================
    
    # display images ==========================================================
    if(DISPLAY_VIDEO):
        cv2.imshow('framecopy', framecopy)
        cv2.imshow('thing', image22)
    #==========================================================================
    
    #saving video
    #out.write(framecopy)
    
    k = cv2.waitKey(30) & 0xff
    if k == 27:
        break
    
    millis2 = time.time()
    millis=millis2 - millis1
    timeslist.append(millis*1000)

cap.release()
#out.release()
cv2.destroyAllWindows()

mean = sum(timeslist)/len(timeslist)