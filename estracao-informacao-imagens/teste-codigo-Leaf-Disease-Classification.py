
import os
import cv2
import numpy as np
import matplotlib.pyplot as plt

#-----Vai fazer a leitura das pastas e as rinderizar em um grafico


dataset = 'img/' #endereço da pasta principal contendo as passas segundarias, a variavel "dataset" armazena o caminho da base de dados
folders = os.listdir(dataset) #o comando "os.listdir()" ira retornar uma lista de todos os nomes dos conteudos contido na pasta endicada, nestecaso o "dataset" que é iqual a "img/", destaca também que esta lista não segue uma organização previa
folders.sort() #como a função anterior não organiza, fui usado a função ".sort()" para ordenar os elementos da liste em ordem cressente
# Count no.of images w.r.t each disease
img_count = {}
for folder in folders: #cria uma laço de repitição para percorre a lista de nomes da pasta
    cnt = len(os.listdir(dataset+folder+os.sep)) #comentarios a baixo
    #a função "len()" é utilisada para fazer a contagem em items de objeto, podem ser vetor or string
    #a função "os.listdir()" retorna uma lista de nomes de uma pasta
    img_count[folder] = cnt
# Plotting barplots of no.of leaf images w.r.t each disease
plt.bar(img_count.keys(), img_count.values())
plt.xticks(rotation='vertical')


#------------------------- fazer a leitura de uma imagem e dividila nos tres cainais de cor "LAB"
image = cv2.imread('img/entrada/folha-de-mamao-menor.jpg')

img_lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)

L = img_lab[:, :, 0]
a = img_lab[:, :, 1]
b = img_lab[:, :, 2]
fig, ax = plt.subplots(1,3, figsize=(15,15))
ax[0].imshow(L)
ax[1].imshow(a)
ax[2].imshow(b)

#-------------- vai fazer a retizada do fundo

# K-means clustering in opencv - https://opencv-python-tutroals.readthedocs.io/en/latest/py_tutorials/py_ml/py_kmeans/py_kmeans_opencv/py_kmeans_opencv.html
pixel_vals = b.flatten()
pixel_vals = np.float32(pixel_vals)
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
# Since we are interested in only actual leaf pixels, we choose 2 clusters
# one cluster for actual leaf pixels and other for unwanted background pixels.
K = 2
retval, labels, centers = cv2.kmeans(pixel_vals, K, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
centers = np.uint8(centers)
segmented_data = centers[labels.flatten()]
segmented_image = segmented_data.reshape((b.shape))
pixel_labels = labels.reshape(img_lab.shape[0], img_lab.shape[1])
# displaying segmented image
plt.imshow(segmented_image)

#-------------

# Doing this, some unwanted pixels that are clustered in main cluster can be avoided.
# Ref - https://docs.opencv.org/3.4/d3/dc0/group__imgproc__shape.html#gac2718a64ade63475425558aa669a943a
pixel_labels = np.uint8(pixel_labels)
ret, components = cv2.connectedComponents(pixel_labels, connectivity=8)
plt.imshow(components, cmap='gray')

#---------
indices = []
for i in range(1, ret):
    row, col = np.where(components==i)
    indices.append(max(len(row), len(col)))
component = np.argmax(np.array(indices))
main_component = component+1   #indexing starts from 0, so we increment by 1 to get actual component index
# creating a mask and extracting pixels corresponding to cluster to which leaf belongs.
# 1 for actual leaf pixels and 0 for other pixels
mask = np.where(components==main_component, 1, 0)
B = image[:, :, 0]
G = image[:, :, 1]
R = image[:, :, 2]
# Extract only masked pixels
r = R*mask
g = G*mask
b = B*mask
final_img = np.dstack((r, g, b))
plt.imshow(final_img)

#---------
v_sum = np.sum(mask, axis=0)
h_sum = np.sum(mask, axis=1)
w = np.count_nonzero(v_sum)
h = np.count_nonzero(h_sum)
x_indices = np.where(v_sum != 0)
y_indices = np.where(h_sum != 0)
x = x_indices[0][0]
y = y_indices[0][0]
final_crop_img = final_img[y:y+h-1, x:x+w-1, :]
final_crop_img = np.uint8(final_crop_img)
img = cv2.resize(final_crop_img, (640,480), interpolation=cv2.INTER_AREA)
plt.imshow(img)

plt.show()