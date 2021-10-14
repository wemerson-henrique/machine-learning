import cv2
from matplotlib import pyplot as plt

img2 = cv2.imread("img/imagensDeTesteEntra/banana/folha2.JPG")

canalDecores = [0]
tamanho = [256]
ranges = [0,255]

histograma = cv2.calcHist([img2],canalDecores,None,tamanho,ranges)

plt.hist(img2.ravel(),256,[0,256])


cv2.imshow("img2",img2)
plt.show()
cv2.waitKey(0)