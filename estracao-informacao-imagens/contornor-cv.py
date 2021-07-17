import numpy as np
import cv2

img = cv2.imread ( 'img/entrada/sigatoka.jpg' )
imgray = cv2.cvtColor (img, cv2.COLOR_BGR2GRAY)
suave = cv2.GaussianBlur(imgray, (7, 7), 0)
bordas =  cv2.Canny(suave,100,200)

ret, thresh = cv2.threshold (bordas, 127, 255, 0)
contornos, hierarquia = cv2.findContours (thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
cv2.drawContours (img, contornos, -1, (255,0,255), 3)
#cv2.drawContours (img, contornos, 2, (0,255,0), 3) #para celecionar uma borda em especifico basta subistituir o "-1"
'''cnt = contornos[0]
cv2.drawContours(img, [cnt], 0, (0,255,0), 3)''' #funcão para mostra um contorno em expecifico
print("Numeros de contornos = " + str(len(contornos)))

#----------------------------------------------------
cnt = contornos[0]
img1 = cv2.drawContours(img, [cnt], 0, (0,255,0), 3)
mask = np.zeros(img1.shape[:2],np.uint8)

bgdModel = np.zeros((1,65),np.float64)
fgdModel = np.zeros((1,65),np.float64)

rect = (50,50,450,290)
cv2.grabCut(img,mask,rect,bgdModel,fgdModel,5,cv2.GC_INIT_WITH_RECT)

mask2 = np.where((mask==2)|(mask==0),0,1).astype('uint8')
img = img1*mask2[:,:,np.newaxis]


cv2.imshow("imagem alterada", img1)

#----------------------------------------------------

cv2.imshow('Imagem',img)
cv2.imshow('Imagem gray',imgray)
cv2.imshow('Bordas Canny',bordas)
cv2.waitKey(0)
cv2.destroyAllWindows()