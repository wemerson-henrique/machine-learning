'''import cv2
image = cv2.imread('sigatoka.jpg')

image[30:50, :] = (255, 0, 0)
#   Cria um retangulo azul por toda a largura da imagem,
#   Este código acima cria um retangulo azul a partir da linha 31 até a linha 50 da imagem
#   e ocupa toda a largura disponível, ou seja, todas as colunas.

image[100:150, 50:100] = (0, 0, 255) #Cria um quadrado vermelho

image[:, 200:220] = (0, 255, 255) #Cria um retangulo amarelo por toda a altura da imagem

image[150:300, 250:350] = (0, 255, 0) #Cria um retangulo verde da linha 150 a 300 nas colunas 250 a 350

image[300:400, 50:150] = (255, 255, 0) #Cria um quadrado ciano da linha 150 a 300 nas colunas 250 a 350

image[250:350, 300:400] = (255, 255, 255) #Cria um quadrado branco

image[70:100, 300: 450] = (0, 0, 0) #Cria um quadrado preto
cv2.imshow("Imagem alterada", image)
#cv2.imwrite("alterada.jpg", image) #vai salvar a imagem com as alterações
cv2.waitKey(0)'''

#---------------------------------------------------------------------------------------------------------------------------

'''import numpy as np
import cv2
imagem = cv2.imread('sigatoka.jpg')
vermelho = (0, 0, 255)
verde = (0, 255, 0)
azul = (255, 0, 0)

cv2.line(imagem, (0, 0), (100, 200), verde)
cv2.line(imagem, (300, 200), (150, 150), vermelho, 5)
cv2.rectangle(imagem, (20, 20), (120, 120), azul, 10)
cv2.rectangle(imagem, (200, 50), (225, 125), verde, -1)
(X, Y) = (imagem.shape[1] // 2, imagem.shape[0] // 2)
for raio in range(0, 175, 15):
    cv2.circle(imagem, (X, Y), raio, vermelho)
cv2.imshow("Desenhando sobre a imagem", imagem)
cv2.waitKey(0)'''

#---------------------------------------------------------------------------------------------------------------------------

import numpy as np
import cv2
imagem = cv2.imread('sigatoka.jpg')
fonte = cv2.FONT_HERSHEY_SIMPLEX
cv2.putText(imagem,'OpenCV',(15,65), fonte,
2,(255,255,255),2,cv2.LINE_AA)
cv2.imshow("Ponte", imagem)
cv2.waitKey(0)