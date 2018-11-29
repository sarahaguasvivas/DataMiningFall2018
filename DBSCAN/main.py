#!/usr/bin/env python3
from sklearn.cluster import DBSCAN
import numpy as np
import cv2
import matplotlib.pyplot as plt
import time
from scipy import ndimage

img= cv2.imread('../data/16b.png', cv2.COLOR_BGR2LAB)
img= np.array(img, dtype= np.float)
plt.subplot(2, 1, 1)
plt.title('High Pass Filtered Image')
plt.axis('off')
plt.imshow(img)

blur= cv2.blur(img, (3, 3))
plt.imshow(blur)
blur= np.reshape(blur, (-1, 3))
rows, cols, chs= img.shape
img= np.reshape(img, (-1, 3))
t0= time.time()
clustering= DBSCAN(eps=5, min_samples=300, algorithm='auto', metric='euclidean').fit_predict(blur)
elapsed= time.time()- t0
print(elapsed)
plt.subplot(2,1,2)
plt.title('DBSCAN Image')
plt.imshow(np.reshape(clustering, [rows, cols]))
plt.colorbar()
plt.axis('off')
plt.show()

