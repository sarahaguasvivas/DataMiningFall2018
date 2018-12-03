from sklearn.cluster import DBSCAN
import numpy as np
import cv2

class StreamClustering:
    def __init__(self, frame= None, method= 'DBSCAN', eps= 0.1, min_points=5):
        self.frame= frame
        self.method= method
        self.eps= eps
        self.min_points= min_points

    def DBSCAN(self, frame):
        return np.reshape(DBSCAN(eps=self.eps, min_samples=self.min_points,
                algorithm='auto',metric='euclidean').fit_predict(frame), frame.shape)




