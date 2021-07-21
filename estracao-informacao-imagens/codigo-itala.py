import cv2 as cv
import numpy as np

img = cv.imread('img/entrada/folha-de-mamao-menor.jpg')

# Converter BGR em HSV
hsv = cv.cvtColor (img, cv.COLOR_BGR2HSV)
# define a faixa de cor azul em HSV
lower_blue = np.array ([0, 30, 30])
upper_blue = np.array ([130,255,255])
# Limite a imagem HSV para obter apenas cores azuis
mask = cv.inRange (hsv, lower_blue, upper_blue)
# Bitwise-AND máscara e imagem original
res = cv.bitwise_and (img, img, mask = mask)

cv.imshow('Bordas retira de azul',res)
cv.waitKey(0)