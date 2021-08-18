import cv2
img = cv2.imread('img/entrada/folha-de-mamao-menor.jpg') #A ordem é BGR e não RGB
img2 = img;
print(img)
print (type(img))
#img1 = cv2.imread('img/entrada/folha-de-mamao.jpg', mode='RGB')

# img1 = img[:, :, ::-1]
# print(img1)
# print (type(img1))
#
# img2 = cv2.GaussianBlur(img, (7, 7), 0)

for y in range(0, img2.shape[0]):
 for x in range(0, img2.shape[1]):
  (b, g, r) = img2[y, x]
  if (g > b + r):
   img2[y, x] = (0, 161, 0)
  else:
   img2[y, x] = (0, 0, 0)
  print('Vermelho:', r, 'Verde:', g, 'Azul:', b)



# cv2.imshow("Imagem", img)
# cv2.imshow("Imagem-1", img1)
 cv2.imshow("Imagem-2", img2)
# cv2.waitKey(0)

'''verde_claro = [90, 255, 90] #padrao RGB
verde_escuro = [15, 30, 15] #padrao RGB
observa se que a frenquencia do verde ideal é dada pela função
G => R+B

'''