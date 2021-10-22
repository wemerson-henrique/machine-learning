import cv2
import LeituraFormatacaoGravacaoDasImagens as Sis

imgPrimeiroModelo = cv2.imread("img/imagensDeTestePreparadasParaSigmentacao/banana/imagem5.JPG",cv2.IMREAD_COLOR)
histogramaPrimeiroModelo = cv2.calcHist([imgPrimeiroModelo],[2],None,[256],[0,256])
imgSegundoModelo = cv2.imread("img/imagensDeTestePreparadasParaSigmentacao/banana/imagem19.JPG",cv2.IMREAD_COLOR)
histogramaSegundoModelo = cv2.calcHist([imgSegundoModelo],[2],None,[256],[0,256])
imgTerceiroModelo = cv2.imread("img/imagensDeTestePreparadasParaSigmentacao/banana/imagem1.JPG",cv2.IMREAD_COLOR)
histogramaTerceiroModelo = cv2.calcHist([imgTerceiroModelo],[2],None,[256],[0,256])


folhasComFunfo = []
folhasSomente = []
folhasPlantacao = []

classeArquivos = Sis.trabalhandoAsImagens()
arquivos = classeArquivos.lendoImagensDaPasta("img/imagensDeTestePreparadasParaSigmentacao/banana")

for i in range(len(arquivos)):
    img = cv2.imread(str(arquivos[i]), cv2.IMREAD_COLOR)
    histograma = cv2.calcHist([img], [2], None, [256], [0, 256])

    compara = cv2.compareHist(histograma, histogramaPrimeiroModelo, cv2.HISTCMP_BHATTACHARYYA)
    print("compara: " + str(compara))
    if compara < 0.2:
        folhasComFunfo.append(str(arquivos[i]))

    compara = cv2.compareHist(histograma, histogramaSegundoModelo, cv2.HISTCMP_BHATTACHARYYA)
    print("compara: " + str(compara))
    if compara < 0.2:
        folhasSomente.append(str(arquivos[i]))

    compara = cv2.compareHist(histograma, histogramaTerceiroModelo, cv2.HISTCMP_BHATTACHARYYA)
    print("compara: " + str(compara))
    if compara < 0.2:
        folhasPlantacao.append(str(arquivos[i]))

    print("--------------------------------------------")

print(str(folhasComFunfo))
print(str(folhasSomente))
print(str(folhasPlantacao))
soma = len(folhasComFunfo)+len(folhasSomente)+len(folhasPlantacao)
print("A quantidade total de arquivos é: "+str(len(arquivos)))
print("A quantidade de arquivos selecionados é: "+str(soma))