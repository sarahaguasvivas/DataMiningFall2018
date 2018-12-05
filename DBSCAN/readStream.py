#!/usr/bin/env python3
import numpy as np
import cv2
import DBSCANstream
import matplotlib.pyplot as plt
import time
cap= cv2.VideoCapture('../data/depth.avi')
cap1= cv2.VideoCapture('../data/color.avi')

a= DBSCANstream.StreamClustering()

while(cap.isOpened()):
    ret, frame= cap.read()
    ret1, frame1= cap1.read()
    depth=cv2.cvtColor(frame, cv2.IMREAD_COLOR)
    color= cv2.cvtColor(frame1, cv2.IMREAD_COLOR)
    t0= time.time()
    clustera= a.ourDBSCAN(color, depth, 0.8)
    elapsed= time.time()-t0
    print(elapsed)
    cv2.imshow('frame', depth)
    cv2.imshow('frame1', color)
    if clustera is not None:
        cv2.imshow('clustera', clustera)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cap1.release()
cv2.destroyAllWindows()
