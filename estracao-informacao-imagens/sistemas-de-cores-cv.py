
#-----------------------------------------------------Máscaras---------------------------------------------------------------------

'''Contudo, existem outros espaços de cores como o próprio “Preto e Branco” ou “tons
de cinza”, além de outros coloridos como o L*a*b* e o HSV. Abaixo temos um exemplo de
como ficaria nossa imagem da ponte nos outros espaços de cores.
'''

import cv2
img = cv2.imread('img/entrada/sigatoka.jpg')
#img = cv2.imread('img/entrada/sigatoka3.jpeg')

cv2.imshow("Original", img)
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
cv2.imshow("Gray", gray)
hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
cv2.imshow("HSV", hsv)
lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
cv2.imshow("L*a*b*", lab)
cv2.waitKey(0)

#-----------------------------------------------------Canais da imagem colorida---------------------------------------------------------------------
'''A função ‘split’ faz o trabalho duro separando os canais 3 canais de cores'''

'''import cv2
img = cv2.imread('img/entrada/sigatoka.jpg')
(canalAzul, canalVerde, canalVermelho) = cv2.split(img)
resultado = cv2.merge([canalAzul, canalVerde, canalVermelho])

cv2.imshow("Vermelho", canalVermelho)
cv2.imshow("Verde", canalVerde)
cv2.imshow("Azul", canalAzul)
cv2.imshow("Imagem original", img)
cv2.imshow("Juncão da separação", resultado)
cv2.waitKey(0)'''

#exibir os canais nas cores originais conforme abaixo

'''import numpy as np
import cv2
img = cv2.imread('img/entrada/sigatoka.jpg')
(canalAzul, canalVerde, canalVermelho) = cv2.split(img)
zeros = np.zeros(img.shape[:2], dtype = "uint8")
cv2.imshow("Vermelho", cv2.merge([zeros, zeros,
canalVermelho]))
cv2.imshow("Verde", cv2.merge([zeros, canalVerde, zeros]))
cv2.imshow("Azul", cv2.merge([canalAzul, zeros, zeros]))
cv2.imshow("Original", img)
cv2.waitKey(0)'''
