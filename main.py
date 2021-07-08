import cv2 as cv
import os
import mediapipe
import time
import numpy as np
import handTracking

detector = handTracking.handDetector(detectionConf=0.85)

# getting images for the header
imagesFolder = "./assets"
imagesList = os.listdir(imagesFolder)
imagesOverlayed = []
for img_ in imagesList:
    my_img = cv.imread(f'{imagesFolder}/{img_}')
    imagesOverlayed.append(my_img)
drawColor = (0,0,255)
lengthBrush = 15
xp,yp = 0,0
imgCanvas = np.zeros((480,640,3),np.uint8)
# start webcam
webcam = cv.VideoCapture(0)
webcam.set(cv.CAP_PROP_FRAME_WIDTH, 640)
webcam.set(cv.CAP_PROP_FRAME_HEIGHT, 480)

header = imagesOverlayed[3]

while True:
    succ , img = webcam.read()
    img = cv.flip(img,1)
    # find hand landmark
    img = detector.findHands(img)
    lmList = detector.findPosition(img)
    if len(lmList) != 0:
        x1,y1 = lmList[8][1:]
        x2,y2 = lmList[12][1:]
        # check up fingers.
        fingers = detector.fingersUp()
        # selection mode.
        if fingers[1] and fingers[2]:
            xp,yp = 0,0
            cv.rectangle(img,(x1,y1 -25),(x2,y2 + 25),drawColor,cv.FILLED)
            if y2 < 50:
                # select color or clean
                if 271<x2<271+60:
                    header = imagesOverlayed[3]
                    drawColor = (0,0,255)
                elif 355<x2<355+60:
                    header = imagesOverlayed[2]
                    drawColor = (0,255,0)
                elif 411<x2<411+60:
                    header = imagesOverlayed[1]
                    drawColor = (255,0,0)
                elif 560<x2<560+60:
                    header = imagesOverlayed[0]
                    drawColor = (0,0,0)
            print('Selection Mode')
        # draw mode.
        if fingers[1] and fingers[2]==False:
            cv.circle(img,(x1,y1),15,drawColor,cv.FILLED)
            if xp == 0 and yp == 0:
                xp,yp = x1,y1
            if drawColor == (0,0,0):
                cv.line(img,(xp,yp),(x1,y1),drawColor,lengthBrush*4)
                cv.line(imgCanvas,(xp,yp),(x1,y1),drawColor,lengthBrush*4)
            else:
                cv.line(img,(xp,yp),(x1,y1),drawColor,lengthBrush)
                cv.line(imgCanvas,(xp,yp),(x1,y1),drawColor,lengthBrush)
            xp,yp = x1,y1
            print('Draw Mode')
    imgGray_ = cv.cvtColor(imgCanvas,cv.COLOR_BGR2GRAY)
    _,imgInv = cv.threshold(imgGray_,50,255,cv.THRESH_BINARY_INV)
    imgInv = cv.cvtColor(imgInv,cv.COLOR_GRAY2BGR)
    img = cv.bitwise_and(img,imgInv)
    img = cv.bitwise_or(img,imgCanvas)
    img[0:50,0:640,:] = header
    cv.imshow('WebCam',img)
    cv.waitKey(1)











