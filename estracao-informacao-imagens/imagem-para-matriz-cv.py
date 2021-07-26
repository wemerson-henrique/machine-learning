import cv2
img = cv2.imread('img/entrada/sigatoka.jpg') #A ordem é BGR e não RGB
print(img)
print (type(img))
#img1 = cv2.imread('img/entrada/folha-de-mamao.jpg', mode='RGB')

img1 = img[:, :, ::-1]

print(img1)
print (type(img1))
cv2.imshow("Imagem", img)
cv2.imshow("Imagem-1", img1)
cv2.waitKey(0)