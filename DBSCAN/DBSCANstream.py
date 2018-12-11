from sklearn.cluster import DBSCAN
import numpy as np
import cv2

class StreamClustering:
    def __init__(self, frame= None, method= 'DBSCAN', eps= 5, min_points=300, algorithm='auto', metric='euclidean'):
        self.frame= frame
        self.method= method
        self.eps= eps
        self.metric= metric
        self.algorithm= algorithm
        self.min_points= min_points

    def skDBSCAN(self, frame):
        if frame is not None:
            frame= np.array(frame, dtype=np.float)
            blur= cv2.blur(frame, (3,3))
            cv2.imshow('blur', blur)
            blur= np.reshape(blur, (-1, 3))
            rows= frame.shape[0]
            cols= frame.shape[1]
            return np.reshape(DBSCAN(eps=self.eps, min_samples=self.min_points,
                    algorithm=self.algorithm,metric=self.metric).fit_predict(blur), [rows, cols])
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

            return np.reshape(DBSCAN(eps=self.eps, min_samples=self.min_points,
                    algorithm=self.algorithm,metric=self.metric).fit_predict(img.reshape((-1,3))), [rows, cols])

        else:
            return None


    def parDBSCAN(self, frame, proportion):
        # Pick a randomly chosen number of points in each row of the images
        # Perform the DBSCAN on the subset of the image with the randomly selected points
        # Perform frequent pattern mining on resulting points
        if frame is not None:
            rows, cols, chs = frame.shape
            newCols= int(cols*proportion)
            newRows= int(rows*proportion)
            redCh= frame[:, :, 0]
            greenCh= frame[:, :, 1]
            blueCh= frame[:, :, 2]
            newImg= np.zeros((newRows, newCols, chs))
            candRows= np.sort(np.random.randint(rows, size=newRows))
            for i in range(newRows):
                candCols= np.sort(np.random.randint(cols, size= newCols))
                bluechannel= blueCh[candRows[i], :]
                redchannel= redCh[candRows[i], :]
                greenchannel= greenCh[candRows[i], :]

                newImg[i, :, 0] = redchannel[candCols]
                newImg[i, :, 1] = greenchannel[candCols]
                newImg[i, :, 2] = bluechannel[candCols]

            newImg= np.array(newImg, dtype=np.float)
            newImg= cv2.blur(newImg, (5, 5))
            clusters= np.reshape(DBSCAN(eps=self.eps, min_samples=self.min_points,
                                algorithm=self.algorithm, metric= self.metric).fit_predict(newImg.reshape((-1, 3))), [newRows, newCols])
            numClusters= len(np.unique(clusters))
            return clusters, numClusters
        else:
            return None
