import cv2 as cv
import os
import mediapipe as mp
import time
import numpy as np


class handDetector():
    def __init__(self, mode=False,maxHands=2,detectionConf = 0.5,trackConf=0.5):
        self.mode = mode
        self.maxHands = maxHands
        self.detectionConf = detectionConf
        self.trackConf = trackConf
        self.mpHands = mp.solutions.hands
        self.hands = self.mpHands.Hands(self.mode,self.maxHands,self.detectionConf,self.trackConf)
        self.mpDraw = mp.solutions.drawing_utils
        self.tipIds = [4,8,12,16,20]
    def findHands(self,img,draw=False):
        imgRGB = cv.cvtColor(img,cv.COLOR_BGR2RGB)
        self.results = self.hands.process(imgRGB)
        if self.results.multi_hand_landmarks:
            for handLands in self.results.multi_hand_landmarks:
                if draw:
                    self.mpDraw.draw_landmarks(img,handLands,self.mpHands.HAND_CONNECTIONS)
        return img
    def findPosition(self,img,handNbr = 0,draw = False):
        self.lmList = []
        if self.results.multi_hand_landmarks:
            myHand = self.results.multi_hand_landmarks[handNbr]
            for id_,lm_ in enumerate(myHand.landmark):
                h,w,c = img.shape
                cx,cy = int(lm_.x*w),int(lm_.y*h)
                self.lmList.append([id_,cx,cy])
                if draw:
                    cv.circle(img,(cx,cy),15,(255,0,255),cv.FILLED)
        return self.lmList
    def fingersUp(self):
        fingers = []
        # Thumb
        if self.lmList[self.tipIds[0]][1] < self.lmList[self.tipIds[0] - 1][1]:
            fingers.append(1)
        else:
            fingers.append(0)
        # fingers
        for id in range(1,5):
            if self.lmList[self.tipIds[id]][2] < self.lmList[self.tipIds[id] -2][2]:
                fingers.append(1)
            else:
                fingers.append(0)
        return fingers



def main():
    pTime = 0
    cTime = 0
    webcam = cv.VideoCapture(0)
    detector = handDetector()
    while True:
        succ , img = webcam.read()
        img = detector.findHands(img)
        lmList = detector.findPosition(img)
        if len(lmList) != 0:
            print(lmList[0])
        cTime = time.time()
        fps = 1/(cTime-pTime)
        pTime = cTime
        cv.putText(img,str(int(fps)),(10,70),cv.FONT_HERSHEY_PLAIN,3,(0,0,255),3)
        cv.imshow('WebCam',img)
        cv.waitKey(1)

if __name__ == '__main__':
    main()