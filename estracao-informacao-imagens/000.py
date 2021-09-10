import cv2
import numpy as np
#-------------Imagens de Teste-------------------------------------
img = cv2.imread('img/entrada/folha-de-mamao-menor.jpg')

img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
img_hsv = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2HSV)

H = img_hsv[:, :, 0]
S = img_hsv[:, :, 1]
G = img_rgb[:, :, 1]
img_hsv_gaussian = cv2.GaussianBlur (S, (5,5), 0)
ret3, th3 = cv2.threshold (img_hsv_gaussian, 0,255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)


image = img_hsv
mask = np.zeros(image.shape[:2], dtype="uint8")
cv2.rectangle(mask, (0, 90), (290, 450), 255, -1) #"mask"= endica em qual imagem sera aplicado, (0, 90), (290, 450), 255, 5)
masked = cv2.bitwise_and(image, image, mask=mask) #"image", image, mask=mask

cv2.imshow("Rectangular Mask", mask)
cv2.imshow("Mask Applied to Image", masked)

cv2.imshow("img_hsv", img_hsv)
cv2.imshow("resultado", th3)

cv2.waitKey(0)