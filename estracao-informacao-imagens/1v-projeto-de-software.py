import cv2
from glob import glob
import indentificacaoDaDoenca2 as IndDoen

arquivos = sorted(glob('img/mamao/*.jpg'))
for i in range(len(arquivos)):
    print(str(arquivos[0]))
    imagem = cv2.imread(arquivos[i])
    cv2.imshow('Imagem', imagem)
    cv2.waitKey(0)
#img1 = IndDoen.Detecta_Doenca(img)
#cv2.imshow("Doença RGB", img.ConvercaoYCrCb())
#cv2.waitKey(0)

imagem1 = cv2.imread("img/entrada/folha-de-mamao-sem-fundo.jpg")

'''img1 = IndDoen.Detecta_Doenca(imagem1)
img1.Convercao_Cores()
img1.Segmentar()
img1.Aplicando_Mascara()'''