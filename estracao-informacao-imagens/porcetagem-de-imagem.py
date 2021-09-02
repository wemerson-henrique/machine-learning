import numpy as np
import cv2
from matplotlib import pyplot as plt

'''img1 = cv2.imread('img/entrada/porcetagem1.png')
img2 = cv2.imread('img/entrada/porcetagem2.png')
info1 = img1.shape
print(info1)
info2 = img2.shape
print(info2)'''



imagem = cv2.imread('img/entrada/porcetagem3.png') #seleciona a imagem

cont = 0
ver = 0

# os for iram percorre a igagem identificando cada pixel
for y in range(0, imagem.shape[0]):
    for x in range(0, imagem.shape[1]):
        (b, g, r) = imagem[y, x]
        if b == 255 and g == 255 and r == 255:
            cont = cont + 1
        else:
            imagem[y, x] = (255, 0, 0)
            ver = ver + 1

altura, largura, canal = imagem.shape
tamanho = altura * largura
print("Numero de pix azuis: ", ver)
print("Numero de pix brancos: ", cont)
pv = (cont / tamanho) * 100
print("A pocentagem ocupada da imagem é: ", pv)

cv2.imshow("Imagem modificada", imagem) #abre uma tela coma a imagem
cv2.waitKey(0) #espera pressionar qualquer tecla