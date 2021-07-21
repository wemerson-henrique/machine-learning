
#-----------------------------------------------------Cortando uma imagem / Crop----------------------------------------------------------------------

'''import cv2
imagem = cv2.imread('sigatoka.jpg')
recorte = imagem[100:200, 100:200]
cv2.imshow("Recorte da imagem", recorte)
cv2.waitKey(0) #espera pressionar qualquer tecla
#cv2.imwrite("recorte.jpg", recorte) #salva no disco'''

'''Usando a mesma imagem ponte.jpg dos exemplos anteriores, temos o resultado abaixo
que é da linha 101 até a linha 200 na coluna 101 até a coluna 200:
'''

#-----------------------------------------------------Redimensionamento / Resize----------------------------------------------------------------------

'''import numpy as np
import cv2
img = cv2.imread('sigatoka.jpg')
cv2.imshow("Original", img) #vai abrir a imagem original
largura = img.shape[1] #largura da imagem
altura = img.shape[0] #altura da imagem
proporcao = float(altura/largura) #calcula a proporção da imagem
largura_nova = 320 #em pixels, cria o novo valor de lagura para a imagem
altura_nova = int(largura_nova*proporcao) #calcula a nova altura da imagem
tamanho_novo = (largura_nova, altura_nova) #atribui os novos valores de largura e altura
img_redimensionada = cv2.resize(img, tamanho_novo, interpolation = cv2.INTER_AREA) #função para criar uma nova imagem com tamanho diferente
cv2.imshow('Resultado', img_redimensionada) #vai abrir a imagem criada na tela
cv2.waitKey(0)'''

#-----------------------------------------------------imagem interpolando linhas---------------------------------------------------------------------

'''O código basicamente refaz a imagem interpolando linhas e colunas, ou seja, pega a
primeira linha, ignora a segunda, depois pega a terceira linha, ignora a quarta, e assim por
diante. '''

'''import numpy as np
import imutils
import cv2
img = cv2.imread('sigatoka.jpg')
cv2.imshow("Original", img)
img_redimensionada = img[::2,::2]
cv2.imshow("Imagem redimensionada", img_redimensionada)
cv2.waitKey(0)'''

#-----------------------------------------------------4.3 Espelhando uma imagem / Flip---------------------------------------------------------------------

#utilisando a bibliotema do openCV para espelhar a imagem

'''import cv2
img = cv2.imread('sigatoka.jpg')
cv2.imshow("Original", img)'''

#Para espelhar uma imagem, basta inverter suas linhas, suas colunas ou ambas.
#Invertendo as linhas temos o flip horizontal e invertendo as colunas temos o flip vertical.
# Nesse caso foi feito a manipulação direta das matrizes que compõe a imagem para fazer o espelhamento

'''flip_horizontal = img[::-1,:] #comando equivalente abaixo
#flip_horizontal = cv2.flip(img, 1)
cv2.imshow("Flip Horizontal", flip_horizontal)
flip_vertical = img[:,::-1] #comando equivalente abaixo
#flip_vertical = cv2.flip(img, 0)
cv2.imshow("Flip Vertical", flip_vertical)
flip_hv = img[::-1,::-1] #comando equivalente abaixo
#flip_hv = cv2.flip(img, -1)
cv2.imshow("Flip Horizontal e Vertical", flip_hv)
cv2.waitKey(0)'''

#-----------------------------------------------------Rotacionando uma imagem / Rotate---------------------------------------------------------------------

'''import cv2
img = cv2.imread('sigatoka.jpg')
(alt, lar) = img.shape[:2] #captura altura e largura
centro = (lar // 2, alt // 2) #acha o centro
M = cv2.getRotationMatrix2D(centro, 30, 1.0) #30 graus
img_rotacionada = cv2.warpAffine(img, M, (lar, alt))
cv2.imshow("Imagem rotacionada em 45 graus", img_rotacionada)
cv2.waitKey(0)'''

#-----------------------------------------------------Máscaras---------------------------------------------------------------------

'''Primeiro é importante definir que uma máscara nada mais é que uma imagem onde
cada pixel pode estar “ligado” ou “desligado”, ou seja, a máscara possui pixels pretos e
brancos apenas. '''

import cv2
import np
img = cv2.imread('sigatoka.jpg')
cv2.imshow("Original", img)
mascara = np.zeros(img.shape[:2], dtype = "uint8")
(cX, cY) = (img.shape[1] // 2, img.shape[0] // 2)
cv2.circle(mascara, (cX, cY), 100, 255, -1)
img_com_mascara = cv2.bitwise_and(img, img, mask = mascara)
cv2.imshow("Máscara aplicada à imagem", img_com_mascara)
cv2.waitKey(0)

#-----------------------------------------------------Máscaras---------------------------------------------------------------------

'''import cv2
import numpy as np
img = cv2.imread('sigatoka.jpg')
cv2.imshow("Original", img)
mascara = np.zeros(img.shape[:2], dtype = "uint8")
(cX, cY) = (img.shape[1] // 2, img.shape[0] // 2)
cv2.circle(mascara, (cX, cY), 180, 255, 70)
cv2.circle(mascara, (cX, cY), 70, 255, -1)
cv2.imshow("Máscara", mascara)
img_com_mascara = cv2.bitwise_and(img, img, mask = mascara)
22
cv2.imshow("Máscara aplicada à imagem", img_com_mascara)
cv2.waitKey(0)'''