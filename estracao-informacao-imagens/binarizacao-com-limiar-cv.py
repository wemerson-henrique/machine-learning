#-----------------------------------------------------8 Binarização com limiar-------------------------------------------------------

'''Thresholding pode ser traduzido por limiarização e no caso de processamento de
imagens na maior parte das vezes utilizamos para binarização da imagem. Normalmente
convertemos imagens em tons de cinza para imagens preto e branco onde todos os pixels
possuem 0 ou 255 como valores de intensidade.'''

'''import numpy as np
import cv2

img = cv2.imread('img/entrada/sigatoka.jpg')
img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
suave = cv2.GaussianBlur(img, (7, 7), 0) # aplica blur
(T, bin) = cv2.threshold(suave, 160, 255, cv2.THRESH_BINARY)
(T, binI) = cv2.threshold(suave, 160, 255,
cv2.THRESH_BINARY_INV)
resultado = np.vstack([
np.hstack([suave, bin]),
np.hstack([binI, cv2.bitwise_and(img, img, mask = binI)])
])
cv2.imshow("Binarização da imagem", resultado)
cv2.waitKey(0)'''

#-----------------------------------------------------8.1 Threshold adaptativo-------------------------------------------------------

import numpy as np
import cv2

img = cv2.imread('img/entrada/sigatoka.jpg')
img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) # converte
suave = cv2.GaussianBlur(img, (7, 7), 0) # aplica blur
bin1 = cv2.adaptiveThreshold(suave, 255,
cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, 21, 5)
bin2 = cv2.adaptiveThreshold(suave, 255,
cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV,
 21, 5)
resultado = np.vstack([
np.hstack([img, suave]),
np.hstack([bin1, bin2])
])
cv2.imshow("Binarização adaptativa da imagem", resultado)
cv2.waitKey(0)

#-----------------------------------------------------8.2 Threshold com Otsu e Riddler-Calvard-------------------------------------------------------

#falha ao importar marotas

'''import mahotas
import numpy as np
import cv2
img = cv2.imread('ponte.jpg')
img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) # converte
suave = cv2.GaussianBlur(img, (7, 7), 0) # aplica blur
T = mahotas.thresholding.otsu(suave)
temp = img.copy()
temp[temp > T] = 255
temp[temp < 255] = 0
temp = cv2.bitwise_not(temp)
T = mahotas.thresholding.rc(suave)
temp2 = img.copy()
temp2[temp2 > T] = 255
temp2[temp2 < 255] = 0
temp2 = cv2.bitwise_not(temp2)
resultado = np.vstack([
np.hstack([img, suave]),
np.hstack([temp, temp2])
])
cv2.imshow("Binarização com método Otsu e Riddler-Calvard",
resultado)
cv2.waitKey(0)'''
