#!/usr/bin/env python3
import numpy as np
import cv2
import DBSCANstream
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import time

cap= cv2.VideoCapture('../data/depth.avi')
cap1= cv2.VideoCapture('../data/color.avi')
SCALE= 0.10
a= DBSCANstream.StreamClustering()
ims=[]
fig=plt.figure()
#try:
while(cap.isOpened()):
    ret, frame= cap.read()
    ret1, frame1= cap1.read()
    depth=cv2.cvtColor(frame, cv2.IMREAD_COLOR)
    color= cv2.cvtColor(frame1, cv2.IMREAD_COLOR)
    t0= time.time()
    #clustera= a.ourDBSCAN(depth)
    #clustera = a.skDBSCAN(depth)
    clustera, numClu, Col, Row= a.parDBSCAN(color, SCALE)
    elapsed= time.time()-t0
    mapped= a.mapBack(clustera, color,depth, Col, Row)
    cv2.imshow('mapped', mapped)
    print(elapsed, numClu)
    cv2.imshow('frame', depth)
    #cv2.imshow('frame1', color)
    if clustera is not None:
        im= plt.imshow(clustera)
        ims.append([im])
        plt.pause(0.0000001)
        plt.draw()
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

#except Exception as e:
#    print(str(e))
  #  print('saving')
  #  ani = animation.ArtistAnimation(fig, ims, interval=50, blit=True,
#                                    repeat_delay=1000)
#    writer= FFMpegWriter(fps=30, metadata= dict(artist='Me'),bitrate=180)

  #  ani.save('DBSCAN.mp4') #, writer=writer)

cap.release()
cap1.release()
cv2.destroyAllWindows()
