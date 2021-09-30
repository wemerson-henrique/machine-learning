import cv2
import numpy as np
import matplotlib.pyplot as plt

# read the image
image = cv2.imread("img/entrada/folha-de-mamao-sem-fundo.jpg")
# convert to RGB
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
# reshape the image to a 2D array of pixels and 3 color values (RGB)
pixel_values = image.reshape((-1, 3))
# convert to float
pixel_values = np.float32(pixel_values)
print(pixel_values.shape)
# define stopping criteria
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.2)
# number of clusters (K)
k = 5
_, labels, (centers) = cv2.kmeans(pixel_values, k, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
# convert back to 8 bit values
centers = np.uint8(centers)

# flatten the labels array
labels = labels.flatten()
# convert all pixels to the color of the centroids
segmented_image = centers[labels.flatten()]
# reshape back to the original image dimension
segmented_image = segmented_image.reshape(image.shape)
# show the image
#plt.imshow(segmented_image)
#plt.show()

masked_image = np.copy(image)
# convert to the shape of a vector of pixel values
masked_image = masked_image.reshape((-1, 3))
# color (i.e cluster) to disable
cluster = 1
masked_image[labels != cluster] = [0, 0, 0]
# convert back to original shape
masked_image = masked_image.reshape(image.shape)
# show the image
plt.imshow(masked_image)
plt.show()

#detectando circulo

img = masked_image
imagemOrinal = img;
cv2.imshow('imagemOrinal',imagemOrinal)
imgray = cv2.cvtColor (img, cv2.COLOR_BGR2GRAY) #deixa preto em branco
suave = cv2.GaussianBlur(imgray, (7, 7), 0) #metodo para fazer a suavização
bordas =  cv2.Canny(suave,100,200) #metodo de para retirar as borads

ret, thresh = cv2.threshold (bordas, 127, 255, 0) #detodo bara indentificar as bordas
contornos, hierarquia = cv2.findContours (thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE) #detodo bara indentificar as bordas

#--------------------- Dezenhando a borda -------------------------------
#cv2.drawContours (img, contornos, -1, (255,0,255), 3)
cv2.drawContours (img, contornos, 2, (0,255,0), 3) #para celecionar uma borda em especifico basta subistituir o "-1"
'''cnt = contornos[0]
cv2.drawContours(img, [cnt], 0, (0,255,0), 3)''' #funcão para mostra um contorno em expecifico
print("Numeros de contornos = " + str(len(contornos)))



cv2.imshow('Imagem',img)
cv2.imshow('Imagem gray',imgray)
cv2.imshow('Bordas Canny',bordas)
cv2.waitKey(0)

'''import cv2
import numpy as np
import matplotlib.pyplot as plt
import sys

# load the image
img = cv2.imread("img/entrada/folha-de-mamao-sem-fundo.jpg")

# convert BGR to RGB to be suitable for showing using matplotlib library
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
# make a copy of the original image
cimg = img.copy()
# convert image to grayscale
img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
# apply a blur using the median filter
img = cv2.medianBlur(img, 5)
# finds the circles in the grayscale image using the Hough transform
circles = cv2.HoughCircles(image=img, method=cv2.HOUGH_GRADIENT, dp=0.9, minDist=80, param1=110, param2=39, maxRadius=70)

for co, i in enumerate(circles[0, :], start=1):
	# draw the outer circle
	cv2.circle(cimg, (i[0], i[1]), i[2], (0, 255, 0), 2)
	# draw the center of the circle
	cv2.circle(cimg, (i[0], i[1]), 2, (0, 0, 255), 3)

# print the number of circles detected
print("Number of circles detected:", co)
# save the image, convert to BGR to save with proper colors
# cv2.imwrite("coins_circles_detected.png", cimg)
# show the image
plt.imshow(cimg)
plt.show()'''