import numpy as np
import cv2
import matplotlib.pyplot as plt
from glob import glob

class LerImagens:
    pass

class ExtracaoFundo:
    pass
#---------------------------------------------------------
class ConvercaoCores:
    def __init__(self,imagem):
        self.imagem
    def ConvercaoYCrCb(self):
        imagem = self.imagem
        imagemYCrCb = cv2.cvtColor(imagem, cv2.COLOR_YCrCb2BGR)
        return imagemYCrCb

arquivos = sorted(glob('img/mamao/*.jpg'))
print(str(arquivos[0]))
img = ConvercaoCores(arquivos[0])
cv2.imshow("Doença RGB", img.ConvercaoYCrCb())
cv2.waitKey(0)