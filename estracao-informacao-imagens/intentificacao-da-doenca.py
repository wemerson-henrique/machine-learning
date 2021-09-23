import numpy as np
import cv2
import matplotlib.pyplot as plt

#-------------Imagens de Teste-------------------------------------
#original_image = cv2.imread('img/16/40e9d306-b56d-4066-b081-895eb2cfed1f.jpg')
#original_image = cv2.imread('img/16/06b4eeba-8283-46a8-9582-4fc8fecec4c6.jpg')
#original_image = cv2.imread('img/37/2bbc7e78-bdea-42c9-aca7-3f37610c95d9.jpg')
#original_image = cv2.imread('img/entrada/folha-de-mamao-menor.jpg')
#original_image = cv2.imread('img/entrada/sigatoka.jpg')
#original_image = cv2.imread('img/entrada/sigatoka1.jpeg')
#original_image = cv2.imread('img/entrada/sigatoka3.jpeg')
#original_image = cv2.imread('img/entrada/tomate1.jpg')
original_image = cv2.imread("img/entrada/folha-de-mamao-sem-fundo.jpg")
#original_image = cv2.imread("img/entrada/plantacao-de-bananeira-png-4.jpg")
#original_image = cv2.imread("img/entrada/folha1.jpg")

original_image = cv2.cvtColor(original_image, cv2.COLOR_YCrCb2BGR)

img=cv2.cvtColor(original_image,cv2.COLOR_BGR2RGB)
vectorized = img.reshape((-1,3))
vectorized = np.float32(vectorized)
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
K = 2
attempts=10
ret,label,center=cv2.kmeans(vectorized,K,None,criteria,attempts,cv2.KMEANS_PP_CENTERS)
center = np.uint8(center)
res = center[label.flatten()]
result_image = res.reshape((img.shape))
figure_size = 15
plt.figure(figsize=(figure_size,figure_size))
plt.subplot(1,2,1),plt.imshow(img)
plt.title('Original Image'), plt.xticks([]), plt.yticks([])
plt.subplot(1,2,2),plt.imshow(result_image)
plt.title('Segmented Image when K = %i' % K), plt.xticks([]), plt.yticks([])
plt.show()

#img=cv2.cvtColor(original_image,cv2.COLOR_BGR2RGB)
YCrCb = cv2.cvtColor(original_image, cv2.COLOR_YCrCb2BGR)
#cv2.imshow("YCrCb",YCrCb)
'''HSV = cv2.cvtColor(original_image, cv2.COLOR_BGR2HSV)
cv2.imshow("HSV",HSV)
LAB = cv2.cvtColor(original_image, cv2.COLOR_BGR2LAB)
cv2.imshow("LAB",LAB)
HLS = cv2.cvtColor(original_image, cv2.COLOR_BGR2HLS)
cv2.imshow("HLS",HLS)
cv2.imshow("original_image",original_image)'''

imag = YCrCb
canal1 = imag[:, :, 0]
canal2 = imag[:, :, 1]
canal3 = imag[:, :, 2]
cv2.imshow("canal1",canal1)
cv2.imshow("canal2",canal2)
cv2.imshow("canal3",canal3)
cv2.waitKey(0)