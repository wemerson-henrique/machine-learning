import cv2
import numpy as np

img = cv2.imread('img/entrada/folha-de-mamao.jpg')

# Converter BGR em HSV
hsv = cv2.cvtColor (img, cv2.COLOR_BGR2HSV)
# define a faixa de cor azul em HSV
lower_blue = np.array ([0, 30, 30])
upper_blue = np.array ([130,255,255])
# Limite a imagem HSV para obter apenas cores azuis
mask = cv2.inRange (hsv, lower_blue, upper_blue)
# Bitwise-AND máscara e imagem original
res = cv2.bitwise_and (img, img, mask = mask)


img = res;
#-------------------------------

largura = img.shape[1]
altura = img.shape[0]
proporcao = float(altura/largura)
largura_nova = 640 #em pixels
altura_nova = int(largura_nova*proporcao)

tamanho_novo = (largura_nova, altura_nova)
img_redimensionada = cv2.resize(img,tamanho_novo, interpolation = cv2.INTER_AREA)


#--------------- DEIXANDO A IMAGEM EM PRETO E BRANCO

img1 = img_redimensionada
imgray = cv2.cvtColor (img1, cv2.COLOR_BGR2GRAY)

#--------------- SUAVIZANDO A IMAGEM

suave = cv2.GaussianBlur(imgray, (7, 7), 0)

#--------------- FAZENDO A RETIRADA DAS BORDAS

bordas =  cv2.Canny(suave,100,200)

#--------------- FAZENDO A INDENTIFICAÇÃO DE CADA BORDA

ret, thresh = cv2.threshold (bordas, 127, 255, 0)
contornos, hierarquia = cv2.findContours (thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

#--------------- DEZENHANDO AS DORDAS INDENTIFICADAS NA IMAGEM URIGINAL

cv2.drawContours (img1, contornos, -1, (255,0,255), 3)
#cv2.drawContours (img, contornos, 2, (0,255,0), 3) #para celecionar uma borda em especifico basta subistituir o "-1"
'''cnt = contornos[0]
cv2.drawContours(img, [cnt], 0, (0,255,0), 3)''' #funcão para mostra um contorno em expecifico

#--------------- CONTA QUANTAS BORDAS FOI INDENTIFICADO E MOSTRA NO CONSOLI

print("Numeros de contornos = " + str(len(contornos)))

#--------------- FAZ A RETIRADA DA IMAGEM

cnt = contornos[0]
img2 = cv2.drawContours(img1, [cnt], 0, (0,255,0), 3)
mask = np.zeros(img1.shape[:2],np.uint8)

bgdModel = np.zeros((1,65),np.float64)
fgdModel = np.zeros((1,65),np.float64)

rect = (50,50,450,290)
cv2.grabCut(img1,mask,rect,bgdModel,fgdModel,5,cv2.GC_INIT_WITH_RECT)

mask2 = np.where((mask==2)|(mask==0),0,1).astype('uint8')
img1 = img2*mask2[:,:,np.newaxis]


cv2.imshow("imagem alterada", img2)
#----------------------------------------------------


#----------------------------------------------------

cv2.imshow('Imagem',img1)
cv2.imshow('Imagem gray',imgray)
cv2.imshow('Bordas Canny',bordas)
cv2.waitKey(0)