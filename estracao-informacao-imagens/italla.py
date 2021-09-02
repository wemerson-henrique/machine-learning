
import cv2 as cv
import matplotlib.pyplot as plt
import numpy as np

img = cv.imread('img/entrada/plantacao-de-bananeira-png-1.jpg')
rows, cols, ch = img.shape
pts1 = np.float32([[56, 65], [368, 52], [28, 387], [389, 390]])
pts2 = np.float32([[0, 0], [300, 0], [0, 300], [300, 300]])
M = cv.getPerspectiveTransform(pts1, pts2)
dst = cv.warpPerspective(img, M, (190, 190))
plt.subplot(121), plt.imshow(img), plt.title('Input')
plt.subplot(122), plt.imshow(dst), plt.title('Output')
cv.imshow("img original",img)
plt.show()