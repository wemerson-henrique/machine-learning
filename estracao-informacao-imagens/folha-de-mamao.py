import numpy as np
import cv2
img = cv2.imread('img/entrada/folha-de-mamao.jpg')
largura = img.shape[1]
altura = img.shape[0]
proporcao = float(altura/largura)
largura_nova = 640 #em pixels
altura_nova = int(largura_nova*proporcao)
17
tamanho_novo = (largura_nova, altura_nova)
img_redimensionada = cv2.resize(img,tamanho_novo, interpolation = cv2.INTER_AREA)


#---------------

img1 = img_redimensionada
img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
suave = cv2.GaussianBlur(img1, (7, 7), 0)
canny1 = cv2.Canny(suave, 20, 120)
canny2 = cv2.Canny(suave, 70, 200)
resultado = np.vstack([
np.hstack([canny1])
])
cv2.imshow("Detector de Bordas Canny", resultado)
cv2.imshow('Resultado', img_redimensionada)
cv2.waitKey(0)