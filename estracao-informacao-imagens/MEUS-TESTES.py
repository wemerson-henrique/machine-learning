import cv2
from indentificacao-da-doenca2 import Detecta-Doenca

img1 = cv2.imread("img/entrada/folha-de-mamao-sem-fundo.jpg")
folh1 = Detecta_Doenca(img1)
folh1.Convercao_Cores()
print(folh1.Convercao_Cores())