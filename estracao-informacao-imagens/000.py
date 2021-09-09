from matplotlib import pyplot as plt
import numpy as np
import cv2
img = cv2.imread('img/entrada/000.jpeg')
img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
h_eq = cv2.equalizeHist(img)
plt.figure()

plt.hist(h_eq.ravel(), 256, [0,256])
plt.xlim([0, 256])
plt.show()
plt.figure()





plt.hist(img.ravel(), 256, [0,256])
plt.xlim([0, 256])
for y in range(0, h_eq.shape[0]):
    for x in range(0, h_eq.shape[1]):
        (b, g, r) = h_eq[y, x]
        if b != 0 and g != 0 and r != 0:
            h_eq[y, x] = (0, 0, 0)
plt.show()
cv2.imshow("Imagem ", img)
cv2.imshow("Imagem modificada", h_eq)
cv2.waitKey(0)