import numpy as np
import cv2

class converteImagemParaYCrCb:
    def __init__(self,imagem):
        self.imagemYCrCb = cv2.cvtColor(imagem, cv2.COLOR_YCrCb2BGR)

class converteImagemParaRGB:
    def __init__(self,imagem):
        self.imagemRGB = cv2.cvtColor(imagem, cv2.COLOR_BGR2RGB)

class aplicandoKmeansNaImagem:
    def __init__(self, imagem):
        Z = np.float32(imagem.reshape((-1, 3)))
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
        k = 5
        ret, label, center = cv2.kmeans(Z, k, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
        center = np.uint8(center)
        res = center[label.flatten()]
        self.res2 = res.reshape((img.shape))

class converteImagemParaTonsDeCinza:
    def __init__(self,imagem):
        self.imagemEmCinza = cv2.cvtColor(imagem, cv2.COLOR_BGR2GRAY)

class aplicaSuavizacaoNaImagemComMetodoGaussiano:
    def __init__(self,imagem):
        self.imagemSuavizada = cv2.GaussianBlur(imagem, (5, 5), 0)

class aplicaBinarizacaoDaImagemMetodoOtsu:
    def __init__(self,imagem):
        imagem = converteImagemParaTonsDeCinza(imagem).imagemEmCinza
        ret3, self.imagemOtsu = cv2.threshold(imagem, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

class aplicaMacara:
    def __init__(self,imagemOriginal,imagemBinarizada):
        mascara = np.zeros(imagemOriginal.shape[:2], dtype="uint8")
        cv2.rectangle(mascara, (0, 90), (290, 450), 255, -1)  # "mask"= endica em qual imagem sera aplicado, (0, 90), (290, 450), 255, 5)
        self.imagemComMascara = cv2.bitwise_and(imagemOriginal, imagemOriginal, mask=imagemBinarizada)

class aplicandoFuncao: #Função não esta fucionando é preciso verificar
    def __init__(self,imagem):
        imagemEmCinza = converteImagemParaTonsDeCinza(imagem).imagemEmCinza
        print("OK")
        imagemSuavizada = aplicaSuavizacaoNaImagemComMetodoGaussiano(imagemEmCinza).imagemSuavizada
        print("OK")
        imagemBinarizada = aplicaBinarizacaoDaImagemMetodoOtsu(imagemSuavizada).imagemOtsu
        print("OK")
        self.imagemResposta = aplicaMacara(imagem,imagemBinarizada)

img1 = cv2.imread("img/entrada/folha-de-mamao-menor.jpg")
cv2.imshow("img1",img1)
a = aplicaBinarizacaoDaImagemMetodoOtsu(img1).imagemOtsu
cv2.imshow("a",a)
b = aplicaMacara(img1,a).imagemComMascara
cv2.imshow("b", b)
cv2.waitKey(0)