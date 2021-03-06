import numpy as np
import cv2
from matplotlib import pyplot as plt
#--------------------------------------------------redimencionando a imagem---------------------------------------------------

img = cv2.imread('img/entrada/folha-de-mamao.jpg')
largura = img.shape[1]
altura = img.shape[0]
proporcao = float(altura/largura)
largura_nova = 640 #em pixels
altura_nova = int(largura_nova*proporcao)
17
tamanho_novo = (largura_nova, altura_nova)
img_redimensionada = cv2.resize(img,tamanho_novo, interpolation = cv2.INTER_AREA)

#--------------------------------------------------retirando plano de fundo sem tratamento---------------------------------------------------

img = img_redimensionada
img2 = img_redimensionada
mask = np.zeros(img.shape[:2],np.uint8)

bgdModel = np.zeros((1,65),np.float64)
fgdModel = np.zeros((1,65),np.float64)

rect = (50,50,450,290)
cv2.grabCut(img,mask,rect,bgdModel,fgdModel,5,cv2.GC_INIT_WITH_RECT)

mask2 = np.where((mask==2)|(mask==0),0,1).astype('uint8')
img = img*mask2[:,:,np.newaxis]


cv2.imshow("imagem alterada", img)
cv2.imshow("imagem original", img2)
cv2.waitKey(0)