import numpy as np
import cv2
import matplotlib.pyplot as plt



#original_image = cv2.imread("img/entrada/folha-de-mamao-sem-fundo.jpg")
'''original_image = cv2.imread("img/imagensDeTesteSaida/banana-brg/imagem1.jpg")
original_image = cv2.imread("img/imagensDeTesteSaida/banana-brg/imagem2.jpg")
original_image = cv2.imread("img/imagensDeTesteSaida/banana-brg/imagem4.jpg")
original_image = cv2.imread("img/imagensDeTesteSaida/banana-brg/imagem22.jpg")
original_image = cv2.imread("img/entrada/sigatoka.jpg")#k=4, 200 valor para doença em claro
original_image = cv2.imread("img/entrada/sigatoka5.jpg")#k=8, 160 valor para doença em escuro
original_image = cv2.imread("img/entrada/sigatoka6.jpg")#k=8, 132 valor para doença  em escuro'''
original_image = cv2.imread("img/entrada/sigatoka7.1.jpg")#k=13, acima de 160 para claro e abaixo de 80 para escuro/// resulado mais enteressante

'''#-----------------------Imagem YCrCb -------------------------------------
image_YCrCb = cv2.cvtColor(original_image, cv2.COLOR_YCrCb2BGR)
cv2.imshow("Imagem Original",image_YCrCb)

gray = cv2.cvtColor(image_YCrCb, cv2.COLOR_BGR2GRAY)
cv2.imshow("preto e branco",gray)

#Função calcHist para calcular o hisograma da imagem
h = cv2.calcHist([gray], [0], None, [256], [0, 256])
plt.figure()
plt.title("Histograma P&B")
plt.xlabel("Intensidade")
plt.ylabel("Qtde de Pixels")
plt.plot(h)
plt.xlim([0, 256])
plt.show()

ret3, otsu1 = cv2.threshold (gray, 0,255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
cv2.imshow("binarização",otsu1)
image = original_image
mask = np.zeros(image.shape[:2], dtype="uint8")
cv2.rectangle(mask, (0, 90), (290, 450), 255, -1)
masked = cv2.bitwise_and(image, image, mask=otsu1)
cv2.imshow("masked",masked)

ret3, otsu1YCrCb = cv2.threshold (gray, 80,255, cv2.THRESH_BINARY + cv2.THRESH_BINARY)
cv2.imshow("binarização com THRESH_BINARY",otsu1)
image = original_image
mask = np.zeros(image.shape[:2], dtype="uint8")
cv2.rectangle(mask, (0, 90), (290, 450), 255, -1)
masked1 = cv2.bitwise_and(image, image, mask=otsu1YCrCb)
cv2.imshow("masked com THRESH_BINARY",masked1)
#-----------------------Fim Imagem YCrCb -------------------------------------'''



'''#-----------------------Imagem YCrCb com kmeans-------------------------------------
image_YCrCb = cv2.cvtColor(original_image, cv2.COLOR_YCrCb2BGR)
imagem = image_YCrCb
cv2.imshow("Imagem Original",image_YCrCb)

Z = np.float32(imagem.reshape((-1, 3)))
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
k = 6
ret, label, center = cv2.kmeans(Z, k, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
center = np.uint8(center)
res = center[label.flatten()]
res1YCrCb = res.reshape((imagem.shape))
cv2.imshow("Aplicacao do kmeansBRG",res1YCrCb)

gray = cv2.cvtColor(res1YCrCb, cv2.COLOR_BGR2GRAY)
cv2.imshow("preto e branco",gray)

#Função calcHist para calcular o hisograma da imagem
h = cv2.calcHist([gray], [0], None, [256], [0, 256])
plt.figure()
plt.title("Histograma P&B")
plt.xlabel("Intensidade")
plt.ylabel("Qtde de Pixels")
plt.plot(h)
plt.xlim([0, 256])
plt.show()

ret3, otsu1 = cv2.threshold (gray, 0,255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
cv2.imshow("binarização",otsu1)
image = original_image
mask = np.zeros(image.shape[:2], dtype="uint8")
cv2.rectangle(mask, (0, 90), (290, 450), 255, -1)
masked = cv2.bitwise_and(image, image, mask=otsu1)
cv2.imshow("masked",masked)

ret3, otsu1YCrCb = cv2.threshold (gray, 45,255, cv2.THRESH_BINARY + cv2.THRESH_BINARY)
cv2.imshow("binarização com THRESH_BINARY",otsu1YCrCb)
image = original_image
mask = np.zeros(image.shape[:2], dtype="uint8")
cv2.rectangle(mask, (0, 90), (290, 450), 255, -1)
masked1 = cv2.bitwise_and(image, image, mask=otsu1YCrCb)
cv2.imshow("masked com THRESH_BINARY",masked1)
#-----------------------Fim Imagem YCrCb com kmeans-------------------------------------'''


#-----------------------Imagem BRG -------------------------------------

imagem = original_image
Z = np.float32(imagem.reshape((-1, 3)))
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
k = 13
ret, label, center = cv2.kmeans(Z, k, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
center = np.uint8(center)
res = center[label.flatten()]
res2BRG = res.reshape((imagem.shape))
cv2.imshow("Aplicacao do kmeansBRG",res2BRG)

grayBRG = cv2.cvtColor(res2BRG, cv2.COLOR_BGR2GRAY)
cv2.imshow("Preto e branco",grayBRG)

#Função calcHist para calcular o hisograma da imagem
h = cv2.calcHist([grayBRG], [0], None, [256], [0, 256])
plt.figure()
plt.title("Histograma P&B")
plt.xlabel("Intensidade")
plt.ylabel("Qtde de Pixels")
plt.plot(h)
plt.xlim([0, 256])
plt.show()

ret3, otsu1BRG = cv2.threshold (grayBRG, 160,255, cv2.THRESH_BINARY + cv2.THRESH_BINARY)
cv2.imshow("binarizacao doente",otsu1BRG)
image = original_image
mask = np.zeros(image.shape[:2], dtype="uint8")
cv2.rectangle(mask, (0, 90), (290, 450), 255, -1)
maskedBRG = cv2.bitwise_and(image, image, mask=otsu1BRG)
cv2.imshow("masked brg doente",maskedBRG)

ret3, otsu2BRG = cv2.threshold (grayBRG, 80,255, cv2.THRESH_BINARY + cv2.THRESH_BINARY_INV)
cv2.imshow("binarizacao saldavel",otsu2BRG)
image = original_image
mask = np.zeros(image.shape[:2], dtype="uint8")
cv2.rectangle(mask, (0, 90), (290, 450), 255, -1)
maskedBRG = cv2.bitwise_and(image, image, mask=otsu2BRG)
cv2.imshow("masked brg saldavel",maskedBRG)

#-----------------------Fim Imagem BRG -------------------------------------
cv2.imshow("original_image",original_image)
#cv2.imshow("image_YCrCb",image_YCrCb)
cv2.waitKey(0)