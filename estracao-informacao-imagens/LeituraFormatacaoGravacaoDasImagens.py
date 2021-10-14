import cv2
from glob import glob

class trabalhandoAsImagens:
    def __init__(self):
        pass
    def lendoImagensDaPasta(self, pasta):
        arquivos = sorted(glob(str(pasta)+'/*.jpg'))
        arquivos = arquivos + sorted(glob(str(pasta)+'/*.png'))
        arquivos = arquivos + sorted(glob(str(pasta)+'/*.jpeg'))
        return arquivos

    def redimencionaImagens(self, arquivos, novaLargura):
        for i in range(len(arquivos)):
            imagem = cv2.imread(arquivos[i])
            alturaImagem, larguraImagem = imagem.shape[0], imagem.shape[1]
            #---
            proporcaoDaImagem = float(alturaImagem / larguraImagem)
            novaAltura = int(novaLargura * proporcaoDaImagem)
            novoTamanho = (novaLargura, novaAltura)
            imagemRedimencionada = cv2.resize(imagem, novoTamanho, interpolation=cv2.INTER_AREA)
            arquivos[i] = imagemRedimencionada
        return arquivos

    def salvaImagens(self, arquivos, caminho):
        for i in range(len(arquivos)):
            cv2.imwrite(str(caminho)+"/imagem" + str(i)+".jpg", arquivos[i])

    def mostraImagens(self,arquivos):
        for i in range(len(arquivos)):
            imagem = cv2.imread(str(arquivos[i]))
            cv2.imshow('Imagem', imagem)
            cv2.waitKey(0)