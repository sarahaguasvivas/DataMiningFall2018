#!/usr/bin/env python3
from sklearn.cluster import DBSCAN
import numpy as np
import cv2
import matplotlib.pyplot as plt
img= cv2.imread('../data/16b.png', cv2.COLOR_BGR2LAB)
img= np.array(img)
plt.subplot(2, 1, 1)
plt.title('Original Image')
plt.axis('off')
plt.imshow(img)

rows, cols, chs= img.shape
img= np.reshape(img, (-1, 3))
clustering= DBSCAN(eps=5, min_samples=300,algorithm='auto', metric='euclidean').fit_predict(img)
plt.subplot(2,1,2)
plt.title('DBSCAN Image')
plt.imshow(np.reshape(clustering, [rows, cols]))
plt.axis('off')
plt.show()

