import numpy as np
import cv2
import matplotlib.pyplot as plt

class Detecta_Doenca:
    def __init__(self,imagem):
        self.imagem = imagem

    def Convercao_Cores(self):
        self.imagem_YCrCb = cv2.cvtColor(self.imagem, cv2.COLOR_YCrCb2BGR)
        self.imagem_rgb = cv2.cvtColor(self.imagem, cv2.COLOR_BGR2RGB)
        cv2.imshow("Imagem Original", self.imagem)
        cv2.imshow("Imagem em YCrCb", self.imagem_YCrCb)
        cv2.imshow("Imagem em RGB", self.imagem_rgb)
        cv2.waitKey(0)

    def Segmentar(self):
        img = self.imagem_rgb
        Z = img.reshape((-1, 3))
        Z = np.float32(Z)
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
        k = 5
        cluster = []

        for i in range(2,k+1):
            ret, label, center = cv2.kmeans(Z, i, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
            center = np.uint8(center)
            self.res_YCrCb = center[label.flatten()]
            self.res2_YCrCb = self.res_YCrCb.reshape((img.shape))
            cv2.imshow("Imagem Original", self.imagem)
            cv2.imshow('Sigmentada do kmeans '+str(i-1), self.res2_YCrCb)
            cv2.waitKey(0)
            cluster.append(self.res2_YCrCb)
            Contorno(self.res2_YCrCb)
            Linha(self.res2_YCrCb)
        #cv2.imshow('res2_YCrCb', cluster[1])
        cv2.waitKey(0)
        #print(cluster)

    def Aplicando_Mascara(self):
        AplicandoMascaraNaImagem(self.res2_YCrCb,self.imagem)

class AplicandoMascaraNaImagem:
    def __init__(self,imagemBinarisada,ImagemOrigem):
        gray = cv2.cvtColor(imagemBinarisada, cv2.COLOR_BGR2GRAY)
        img_hsv_gaussian = cv2.GaussianBlur(gray, (5, 5), 0)
        ret3, otsu1 = cv2.threshold(img_hsv_gaussian, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        cv2.imshow("binarização", otsu1)
        image = ImagemOrigem
        mask = np.zeros(image.shape[:2], dtype="uint8")
        cv2.rectangle(mask, (0, 90), (290, 450), 255, -1)  # "mask"= endica em qual imagem sera aplicado, (0, 90), (290, 450), 255, 5)
        masked = cv2.bitwise_and(image, image, mask=otsu1)  # "image", image, mask=mask
        cv2.imshow("Imagem Original", ImagemOrigem)
        cv2.imshow("Imagem Mascarada", masked)
        cv2.waitKey(0)
        Porcentagem(masked)

class Contorno:
    def __init__(self,imagem):
        cv2.imshow('Imagem do Kmeans', imagem)
        gray = cv2.cvtColor(imagem, cv2.COLOR_BGR2GRAY)
        suave = gray #cv2.GaussianBlur(imgray, (7, 7), 0)
        bordas = cv2.Canny(suave, 100, 200)
        ret, thresh = cv2.threshold(bordas, 127, 255, 0)
        contornos, hierarquia = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        # --------------------- Dezenhando a borda -------------------------------
        #cv2.drawContours(imagem, contornos, -1, (255, 255, 0), 3)
        print("Numeros de contornos = " + str(len(contornos)))
        cv2.imshow('Borda dezenhada', imagem)
        cv2.waitKey(0)

class Linha:
    def __init__(self,imagem):
        gray = cv2.cvtColor(imagem, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, 50, 150, apertureSize=3)
        comprimento_minimo = 1
        intervalo_maximo = 20
        lines = cv2.HoughLinesP(image=edges, rho=1, theta=0.1 * np.pi / 360, threshold=50,
                               minLineLength=comprimento_minimo, maxLineGap=intervalo_maximo)
        print("Numeros de linhas = ",len(lines))
        '''for line in lines:
            x1, y1, x2, y2 = line[0]
            cv2.line(imagem, (x1, y1), (x2, y2), (0, 255, 0), 2)'''
        cv2.imshow('Resultado', imagem)
        cv2.waitKey(0)

class Porcentagem:
    def __init__(self,imagem):
        pixelsAtivos = 0
        pixelsInativos = 0

        # os for iram percorre a igagem identificando cada pixel
        for y in range(0, imagem.shape[0]):
            for x in range(0, imagem.shape[1]):
                (b, g, r) = imagem[y, x]
                if b != 0 and g != 0 and r != 0:
                    pixelsAtivos = pixelsAtivos + 1
                else:
                    pixelsInativos = pixelsInativos + 1

        alturaDaImagem, larguraDaImagem, canaisDeCoresDaImagem = imagem.shape
        tamanhoDaImagem = alturaDaImagem * larguraDaImagem
        print("Tamanho total da imagem em pixels: ", tamanhoDaImagem,
              " altura: ", alturaDaImagem, " largura: ", larguraDaImagem, " canal: ", canaisDeCoresDaImagem)
        print("Numero de pixels brancos/ativos: ", pixelsAtivos)
        print("Numero de pixels pretos/inativos: ", pixelsInativos)
        porcentagemDePixelsAtivos = (pixelsAtivos / tamanhoDaImagem) * 100
        print("A pocentagem ocupada da imagem é: ", porcentagemDePixelsAtivos)

#--------------------------
imagem1 = cv2.imread("img/entrada/folha-de-mamao-sem-fundo.jpg")

img1 = Detecta_Doenca(imagem1)
img1.Convercao_Cores()
img1.Segmentar()
img1.Aplicando_Mascara()