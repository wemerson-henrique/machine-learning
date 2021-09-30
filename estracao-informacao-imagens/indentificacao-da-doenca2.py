import numpy as np
import cv2
import matplotlib.pyplot as plt

class Detecta_Doenca:
    def __init__(self,vetor_folha):
        self.vetor_imagem = vetor_folha

    def Convercao_Cores(self):
        self.image_YCrCb = cv2.cvtColor(self.vetor_imagem, cv2.COLOR_YCrCb2BGR)
        self.image_rgb = cv2.cvtColor(self.vetor_imagem, cv2.COLOR_BGR2RGB)
        cv2.imshow("Imagem Original", self.vetor_imagem)
        cv2.imshow("Imagem YCrCb", self.image_YCrCb)
        cv2.imshow("Doença RGB", self.image_rgb)
        cv2.waitKey(0)

    def Segmentar(self):
        img = self.image_YCrCb
        Z = img.reshape((-1, 3))
        Z = np.float32(Z)
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
        K = 5
        ret, label, center = cv2.kmeans(Z, K, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
        center = np.uint8(center)
        self.res_YCrCb = center[label.flatten()]
        self.res2_YCrCb = self.res_YCrCb.reshape((img.shape))
        cv2.imshow("Imagem Original", self.vetor_imagem)
        cv2.imshow('res2_YCrCb', self.res2_YCrCb)
        cv2.waitKey(0)

    def Aplicando_Mascara(self):
        gray = cv2.cvtColor(self.res2_YCrCb, cv2.COLOR_BGR2GRAY)

        img_hsv_gaussian = cv2.GaussianBlur(gray, (5, 5), 0)
        ret3, otsu1 = cv2.threshold(img_hsv_gaussian, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        cv2.imshow("binarização", otsu1)
        image = self.vetor_imagem
        mask = np.zeros(image.shape[:2], dtype="uint8")
        cv2.rectangle(mask, (0, 90), (290, 450), 255, -1)  # "mask"= endica em qual imagem sera aplicado, (0, 90), (290, 450), 255, 5)
        masked = cv2.bitwise_and(image, image, mask=otsu1)  # "image", image, mask=mask
        cv2.imshow("Imagem Original", self.vetor_imagem)
        cv2.imshow("masked", masked)
        cv2.waitKey(0)

#--------------------------
imagem1 = cv2.imread("img/entrada/folha-de-mamao-sem-fundo.jpg")

img1 = Detecta_Doenca(imagem1)
img1.Convercao_Cores()
img1.Segmentar()
img1.Aplicando_Mascara()