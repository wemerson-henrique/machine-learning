import cv2
import numpy as np
import matplotlib.pyplot as plt
import LeituraFormatacaoGravacaoDasImagens as sisIF


class removeFundo:
    def __init__(self):
        pass

    def divideImagenEmCanaisDeCor(self, imagem):
        L = imagem[:, :, 0]
        a = imagem[:, :, 1]
        b = imagem[:, :, 2]
        return L, a, b

    def retiraFundo(self, b):
        pixel_vals = np.float32(b.flatten())
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
        K = 2
        retval, labels, centers = cv2.kmeans(pixel_vals, K, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
        centers = np.uint8(centers)
        segmented_data = centers[labels.flatten()]
        segmented_image = segmented_data.reshape((b.shape))
        pixel_labels = labels.reshape(img_lab.shape[0], img_lab.shape[1])

        pixel_labels = np.uint8(pixel_labels)
        ret, components = cv2.connectedComponents(pixel_labels, connectivity=8)

        indices = []
        for i in range(1, ret):
            row, col = np.where(components == i)
            indices.append(max(len(row), len(col)))
        component = np.argmax(np.array(indices))
        main_component = component + 1
        mask = np.where(components == main_component, 1, 0)
        B = image[:, :, 0]
        G = image[:, :, 1]
        R = image[:, :, 2]
        r = R * mask
        g = G * mask
        b = B * mask
        final_img = np.dstack((b, g, r))
        return final_img

b = sisIF.trabalhandoAsImagens()
imagensEntrada = b.lendoImagensDaPasta("img/imagensDeTestePreparadasParaSigmentacao/mamao")
print(str(imagensEntrada))

for i in range(len(imagensEntrada)):
    image = cv2.imread(imagensEntrada[i])

img_lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)

removendoFundo = removeFundo()

L, a, b = removendoFundo.divideImagenEmCanaisDeCor(img_lab)
imagemSemFundo = removendoFundo.retiraFundo(b)
plt.imshow(imagemSemFundo)
cv2.imwrite("img/Imagem-exemplo.jpg",imagemSemFundo)

cv2.waitKey(0)
plt.show()