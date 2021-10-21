import cv2
import numpy as np
from matplotlib import pyplot as plt

img1 = cv2.imread("img/imagensDeTestePreparadasParaSigmentacao/banana/imagem8.JPG",cv2.IMREAD_COLOR)
img2 = cv2.imread("img/imagensDeTestePreparadasParaSigmentacao/banana/imagem9.JPG",cv2.IMREAD_COLOR)
histograma1 = cv2.calcHist([img1],[2],None,[256],[0,256])
histograma2 = cv2.calcHist([img2],[2],None,[256],[0,256])


compara = cv2.compareHist(histograma1, histograma2, cv2.HISTCMP_BHATTACHARYYA)

print(str(compara))