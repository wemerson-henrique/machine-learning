import cv2
import LeituraFormatacaoGravacaoDasImagens as Sis
import numpy as np
from matplotlib import pyplot as plt

'''img1 = cv2.imread("img/imagensDeTestePreparadasParaSigmentacao/banana/imagem16.JPG",cv2.IMREAD_COLOR)
img2 = cv2.imread("img/imagensDeTestePreparadasParaSigmentacao/banana/imagem17.JPG",cv2.IMREAD_COLOR)
histograma1 = cv2.calcHist([img1],[2],None,[256],[0,256])
histograma2 = cv2.calcHist([img2],[2],None,[256],[0,256])'''
imgPrimeiroModelo = cv2.imread("img/imagensDeTestePreparadasParaSigmentacao/banana/imagem5.JPG",cv2.IMREAD_COLOR)
histogramaPrimeiroModelo = cv2.calcHist([imgPrimeiroModelo],[2],None,[256],[0,256])
imgSegundoModelo = cv2.imread("img/imagensDeTestePreparadasParaSigmentacao/banana/imagem1.JPG",cv2.IMREAD_COLOR)
histogramaSegundoModelo = cv2.calcHist([imgSegundoModelo],[2],None,[256],[0,256])
imgTerceiroModelo = cv2.imread("img/imagensDeTestePreparadasParaSigmentacao/banana/imagem16.JPG",cv2.IMREAD_COLOR)
histogramaTerceiroModelo = cv2.calcHist([imgTerceiroModelo],[2],None,[256],[0,256])


folhasComFunfo = []
folhasSomente = []
folhasPlantacao = []
melhor = []

classeArquivos = Sis.trabalhandoAsImagens()
arquivos = classeArquivos.lendoImagensDaPasta("img/imagensDeTestePreparadasParaSigmentacao/banana")
print(str(arquivos))

histograma2 = histogramaSegundoModelo

for i in range(len(arquivos)):
    img = cv2.imread(str(arquivos[i]), cv2.IMREAD_COLOR)
    histograma = cv2.calcHist([img], [2], None, [256], [0, 256])

    print("-------------------------------------------"+str(i))
    #maior melhor
    compara = cv2.compareHist(histograma, histograma2, cv2.HISTCMP_CORREL)
    print("compara: "+str(compara))
    if compara > 1.2:
        print("São semelhantes "+str(compara))
        folhasPlantacao.append(str(arquivos[i]))
    else:
        print("Não são semelhantes " + str(compara))
    print("-------------------------------------------")

    #menor melhor
    compara = cv2.compareHist(histograma, histograma2, cv2.HISTCMP_CHISQR)
    print("compara: "+str(compara))
    if compara < 0.2:
        print("São semelhantes "+str(compara))
        folhasSomente.append(str(arquivos[i]))
    else:
        print("Não são semelhantes " + str(compara))
    print("-------------------------------------------")

    #maior melhor
    compara = cv2.compareHist(histograma, histograma2, cv2.HISTCMP_INTERSECT)
    print("compara: "+str(compara))
    if compara > 1.2:
        print("São semelhantes "+str(compara))
        folhasPlantacao.append(str(arquivos[i]))
    else:
        print("Não são semelhantes " + str(compara))
    print("-------------------------------------------")

    #menor melhor
    compara = cv2.compareHist(histograma, histograma2, cv2.HISTCMP_BHATTACHARYYA)
    print("compara: "+str(compara))
    if compara < 0.2:
        print("São semelhantes "+str(compara))
        melhor.append(str(arquivos[i]))
    else:
        print("Não são semelhantes " + str(compara))
    print("-------------------------------------------")

print(str(folhasPlantacao))
print(str(folhasComFunfo))
print(str(folhasSomente))
print(str(melhor))