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

    def DBSCAN(self, frame):

        if frame is not None:
            img= np.array(frame, dtype=np.float)
            blur= cv2.blur(img, (3,3))
            blur= np.reshape(blur, (-1, 3))
            rows, cols, chs= img.shape

            return np.reshape(DBSCAN(eps=self.eps, min_samples=self.min_points,
                    algorithm=self.algorithm,metric=self.metric).fit_predict(blur), [rows, cols])
        else:
            return None


