import numpy as np
import cv2

img = cv2.imread ( 'img/entrada/sigatoka.jpg' )
imagemOrinal = img;
cv2.imshow('imagemOrinal',imagemOrinal)
imgray = cv2.cvtColor (img, cv2.COLOR_BGR2GRAY) #deixa preto em branco
suave = cv2.GaussianBlur(imgray, (7, 7), 0) #metodo para fazer a suavização
bordas =  cv2.Canny(suave,100,200) #metodo de para retirar as borads

ret, thresh = cv2.threshold (bordas, 127, 255, 0) #detodo bara indentificar as bordas
contornos, hierarquia = cv2.findContours (thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE) #detodo bara indentificar as bordas

#--------------------- Dezenhando a borda -------------------------------
#cv2.drawContours (img, contornos, -1, (255,0,255), 3)
cv2.drawContours (img, contornos, 2, (0,255,0), 3) #para celecionar uma borda em especifico basta subistituir o "-1"
'''cnt = contornos[0]
cv2.drawContours(img, [cnt], 0, (0,255,0), 3)''' #funcão para mostra um contorno em expecifico
print("Numeros de contornos = " + str(len(contornos)))



cv2.imshow('Imagem',img)
cv2.imshow('Imagem gray',imgray)
cv2.imshow('Bordas Canny',bordas)

cv2.waitKey(0)