
#-----------------------------------------------------Máscaras---------------------------------------------------------------------

'''Contudo, existem outros espaços de cores como o próprio “Preto e Branco” ou “tons
de cinza”, além de outros coloridos como o L*a*b* e o HSV. Abaixo temos um exemplo de
como ficaria nossa imagem da ponte nos outros espaços de cores.
'''

'''import cv2
img = cv2.imread('sigatoka.jpg')
24
cv2.imshow("Original", img)
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
cv2.imshow("Gray", gray)
hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
cv2.imshow("HSV", hsv)
lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
cv2.imshow("L*a*b*", lab)
cv2.waitKey(0)'''

#-----------------------------------------------------Canais da imagem colorida---------------------------------------------------------------------

import cv2
img = cv2.imread('sigatoka.jpg')
(canalAzul, canalVerde, canalVermelho) = cv2.split(img)
cv2.imshow("Vermelho", canalVermelho)
cv2.imshow("Verde", canalVerde)
cv2.imshow("Azul", canalAzul)
cv2.waitKey(0)
