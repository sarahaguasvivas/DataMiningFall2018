from sklearn.cluster import DBSCAN
import numpy as np
import cv2

class StreamClustering:
    def __init__(self, frame= None, method= 'DBSCAN', eps= 0.1, min_points=300, algorithm='auto', metric='euclidean'):
        self.frame= frame
        self.method= method
        self.eps= eps
        self.metric= metric
        self.algorithm= algorithm
        self.min_points= min_points

    def skDBSCAN(self, frame):
        if frame is not None:
            img= np.array(frame, dtype=np.float)
            blur= cv2.blur(img, (3,3))
            blur= np.reshape(blur, (-1, 3))
            rows, cols, chs= img.shape

            return np.reshape(DBSCAN(eps=self.eps, min_samples=self.min_points,
                    algorithm=self.algorithm,metric=self.metric).fit_predict(blur), [rows, cols])
        else:
            return None

    def ourDBSCAN(self, color, depth, proportionPoints):
        depthLimit= 170
        depthUpper= 175

        if color is not None:

            red=  depth[:, :, 0]
            green=depth[:, :, 1]
            blue= depth[:, :, 2]

            grRed= np.where(np.abs(red)< depthLimit, red, 0)
            grGreen= np.where(np.abs(green)< depthLimit , green, 0)
            grBlue= np.where(np.abs(blue) < depthLimit , blue, 0)

            grRed= np.where(np.abs(red)> depthUpper, red, 0)
            grGreen= np.where(np.abs(green)> depthUpper , green, 0)
            grBlue= np.where(np.abs(blue) > depthUpper , blue, 0)

            img= np.zeros(color.shape)

            img[:, :, 0]= grRed
            img[:, :, 1]= grGreen
            img[:, :, 2]= grBlue

            cv2.imshow('img', img)
        else:
            return None


    def parDBSCAN(self, frame):
        pass
