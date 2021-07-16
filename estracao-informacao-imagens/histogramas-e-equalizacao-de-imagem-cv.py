
#-----------------------------------------------------6 Histogramas e equalização de imagem---------------------------------------------------------------------

'''Um histograma é um gráfico de colunas ou de linhas que representa a distribuição dos
valores dos pixels de uma imagem, ou seja, a quantidade de pixeis mais claros (próximos de
255) e a quantidade de pixels mais escuros (próximos de 0).
O eixo X do gráfico normalmente possui uma distribuição de 0 a 255 que demonstra o
valor (intensidade) do pixel e no eixo Y é plotada a quantidade de pixels daquela intensidade.
'''

'''from matplotlib import pyplot as plt
import cv2
img = cv2.imread('img/entrada/sigatoka.jpg')
img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) #converte P&B
cv2.imshow("Imagem P&B", img)
#Função calcHist para calcular o hisograma da imagem
h = cv2.calcHist([img], [0], None, [256], [0, 256])
plt.figure()
plt.title("Histograma P&B")
plt.xlabel("Intensidade")
plt.ylabel("Qtde de Pixels")
plt.plot(h)
plt.xlim([0, 256])
plt.show()
cv2.waitKey(0)'''

#--------------------------------------------------------------------------------------------------------------------------

'''Também é possível plotar o histograma de outra forma, com a ajuda da função
‘ravel()’. Neste caso o eixo X avança o valor 255 indo até 300, espaço que não existem pixels.'''

'''from matplotlib import pyplot as plt
import cv2
img = cv2.imread('img/entrada/sigatoka.jpg')
img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) #converte P&B
cv2.imshow("Imagem P&B", img)
#Função calcHist para calcular o hisograma da imagem
plt.hist(img.ravel(),256,[0,256])
plt.show() 
h = cv2.calcHist([img], [0], None, [256], [0, 256])
plt.figure()
plt.title("Histograma P&B")
plt.xlabel("Intensidade")
plt.ylabel("Qtde de Pixels")
plt.plot(h)
plt.xlim([0, 256])
plt.show()
cv2.waitKey(0)'''

#--------------------------------------------------------------------------------------------------------------------------

#esse codigo não funcionol fazer verificação

'''from matplotlib import pyplot as plt
import numpy as np
import cv2
img = cv2.imread('img/entrada/sigatoka.jpg')
cv2.imshow("Imagem Colorida", img)
#Separa os canais
canais = cv2.split(img)
cores = ("b", "g", "r")
plt.figure()
plt.title("Histograma Colorido")
plt.xlabel("Intensidade")
plt.ylabel("Número de Pixels")
for (canal, cor) in zip(canais, cores):
    #Este loop executa 3 vezes, uma para cada canal
    hist = cv2.calcHist([canal], [0], None, [256], [0, 256])
    plt.plot(hist, cor = cor)
    plt.xlim([0, 256])
plt.show()'''

#-----------------------------------------------------6.1 Equalização de Histograma---------------------------------------------------------------------

from matplotlib import pyplot as plt
import numpy as np
import cv2
img = cv2.imread('img/entrada/sigatoka.jpg')
img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
h_eq = cv2.equalizeHist(img)
plt.figure()
plt.title("Histograma Equalizado")
plt.xlabel("Intensidade")
plt.ylabel("Qtde de Pixels")
plt.hist(h_eq.ravel(), 256, [0,256])
plt.xlim([0, 256])
plt.show()
plt.figure()
plt.title("Histograma Original")
plt.xlabel("Intensidade")
plt.ylabel("Qtde de Pixels")
plt.hist(img.ravel(), 256, [0,256])
plt.xlim([0, 256])
plt.show()
cv2.imshow("Imagem ", img)
cv2.imshow("Imagem modificada", h_eq)
cv2.waitKey(0)