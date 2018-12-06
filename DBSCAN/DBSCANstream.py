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
            img= np.reshape(frame, [-1, 3])
            rows= frame.shape[0]
            cols= frame.shape[1]

            return np.reshape(DBSCAN(eps=self.eps, min_samples=self.min_points,
                    algorithm=self.algorithm,metric=self.metric).fit_predict(img), [rows, cols])
        else:
            return None

    def ourDBSCAN(self, depth):
        depthLimit= 170
        depthUpper= 175

        if depth is not None:
            rows, cols, _ = depth.shape

            red=  depth[:, :, 0]
            green=depth[:, :, 1]
            blue= depth[:, :, 2]

            grRed= np.where(np.abs(red)< depthLimit, red, 0)
            grGreen= np.where(np.abs(green)< depthLimit , green, 0)
            grBlue= np.where(np.abs(blue) < depthLimit , blue, 0)

            grRed= np.where(np.abs(red)> depthUpper, red, 0)
            grGreen= np.where(np.abs(green)> depthUpper , green, 0)
            grBlue= np.where(np.abs(blue) > depthUpper , blue, 0)

            img= np.zeros(depth.shape)
            img[:, :, 0]= grRed
            img[:, :, 1]= grGreen
            img[:, :, 2]= grBlue
            #cv2.imshow('img', img)
            print(grBlue.shape)
            toCluster= grBlue.reshape([-1, 3])
            return np.reshape(DBSCAN(eps=self.eps, min_samples=self.min_points,
                    algorithm=self.algorithm,metric=self.metric).fit_predict(toCluster), [rows, cols])

        else:
            return None


    def parDBSCAN(self, frame):
        pass
