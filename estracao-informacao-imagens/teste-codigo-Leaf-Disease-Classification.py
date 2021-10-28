
import os
import cv2
import cv2.cv2
import numpy as np
import matplotlib.pyplot as plt

#-----Vai fazer a leitura das pastas e as rinderizar em um grafico


'''dataset = 'img/' #endereço da pasta principal contendo as passas segundarias, a variavel "dataset" armazena o caminho da base de dados
folders = os.listdir(dataset) #o comando "os.listdir()" ira retornar uma lista de todos os nomes dos conteudos contido na pasta endicada, nestecaso o "dataset" que é iqual a "img/", destaca também que esta lista não segue uma organização previa
folders.sort() #como a função anterior não organiza, fui usado a função ".sort()" para ordenar os elementos da liste em ordem cressente
# Count no.of images w.r.t each disease
img_count = {}
for folder in folders: #cria uma laço de repitição para percorre a lista de nomes da pasta
    cnt = len(os.listdir(dataset+folder+os.sep)) #comentarios a baixo
    #a função "len()" é utilisada para fazer a contagem em items de objeto, podem ser vetor or string
    #a função "os.listdir()" retorna uma lista de nomes de uma pasta
    # entenda que: dataset= é a pasta home com as imagens; folder= são os subpastas; os.sep= é um tratamento de caminho para universalizar enpende do sintema operacional "os"
    # basicamente o comando "len(os.listdir(dataset + folder + os.sep))" ira contar todos os intens das subpast da pasta home de imagens, retornar o valor para a variavel "cnt"
    img_count[folder] = cnt #atrui ao vetor "img_count" o nome das subpastas e a quantidades de items que ela possui
    #print(img_count)
#----------------------
# mostrarndo os dados no grafico, respetivo a cada doença
#OBS: estudar mais sobre
plt.bar(img_count.keys(), img_count.values())
plt.xticks(rotation='vertical')
#----------------------'''


#------------------------- fazer a leitura de uma imagem e dividila nos tres cainais de cor "LAB"
#-------------Imagens de Teste-------------------------------------
#image = cv2.imread('img/16/40e9d306-b56d-4066-b081-895eb2cfed1f.jpg')
#image = cv2.imread('img/16/06b4eeba-8283-46a8-9582-4fc8fecec4c6.jpg')
image = cv2.imread('img/37/2bbc7e78-bdea-42c9-aca7-3f37610c95d9.jpg')
#image = cv2.imread('img/entrada/folha-de-mamao-menor.jpg')
#image = cv2.imread('img/entrada/sigatoka.jpg')
#image = cv2.imread('img/entrada/sigatoka1.jpeg')
#image = cv2.imread('img/entrada/sigatoka3.jpeg')
#image = cv2.imread('img/entrada/tomate1.jpg')
#image = cv2.imread("img/entrada/folha-de-mamao-sem-fundo.jpg")
#image = cv2.imread("img/entrada/plantacao-de-bananeira-png-4.jpg")
#image = cv2.imread('img/entrada/folha2.jpg')
#image = cv2.imread('img/entrada/plantacao-de-bananeira-png-3.jpg') #faz a leitura de uma imagem
#image = cv2.imread('img/imagensDeTestePreparadasParaSigmentacao/banana/imagem1.JPG')

img_lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB) #faz a converção da imagem do padão de cor BGR para LAB
#OBS: o numpy trabalha com o canal RGB invertido de modo que fica BGR

#as proximas trez linhas ira dividir o canal de cor LAB da imagem
L = img_lab[:, :, 0] #a variavel "L" ira receber o canal de cor 0 da imagem, ou seja a fatia de corror L do padrão LAB de cores
a = img_lab[:, :, 1] #a variavel "a" ira receber o canal de cor 1 da imagem, ou seja a fatia de corror a do padrão LAB de cores
b = img_lab[:, :, 2] #a variavel "b" ira receber o canal de cor 2 da imagem, ou seja a fatia de corror v do padrão LAB de cores
fig, ax = plt.subplots(1,3, figsize=(15,15)) #vai criar uma figura e um conjunto de subtramas.
# "figsize=(15,15)" => tamanho da figura
# "1,3," => "1" padrão da função, "3" indica o numero de imagens a figura possuira
# "ax" => retorna uma matrix de eixo de  x e y a imagem, que nesse caso so pussuira o eixo x
# "fig" => é a figura criaga
ax[0].imshow(L) # vai adcionar a vaivel L ao eixo no locau "0" da figura
ax[1].imshow(a) # vai adcionar a vaivel a ao eixo no locau "1" da figura
ax[2].imshow(b) # vai adcionar a vaivel b ao eixo no locau "2" da figura

#-------------- vai fazer a retizada do fundo

# K-means clustering in opencv - https://opencv-python-tutroals.readthedocs.io/en/latest/py_tutorials/py_ml/py_kmeans/py_kmeans_opencv/py_kmeans_opencv.html
pixel_vals = b.flatten() #esplicação abaixo
# O comando "b.flatten()" Retorne uma cópia da matriz recolhida em uma dimensão,
# basicamente vai pegar uma matriz e nivela-la para que as matrizes no seu enterior fiquem uma mesma matriz ou nivel,
# Exemplo: [[1,1,1],[2,2,2],[3,3,3]] => [1,1,1,2,2,2,3,3,3]
pixel_vals = np.float32(pixel_vals) #converte os valores da matriz "pixel_vals" para o tipo float
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0) #vai aplicar a configuração inicial para função KMeans, explicação abaixo
# "(cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)" bazicamente esta definindo um criterio de interação e precisão
#critério é tal que, sempre que 10 iterações do algoritmo são executadas,
# ou uma precisão de épsilon = 1,0 é alcançada, pare o algoritmo e retorne a resposta.

# Uma vez que estamos interessados apenas em pixels de folhas reais, escolhemos 2 clusters
# um cluster para pixels de folha reais e outro para pixels de fundo indesejados.
K = 2 #criação de clusters
retval, labels, centers = cv2.kmeans(pixel_vals, K, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS) #explicação a baixo, retirados do opencv documente
#A função "cv2.kmeans()" implementa um algoritmo k-means que encontra os centros dos clusters cluster_count e agrupa as amostras de entrada em torno dos clusters.
# Como saída,bestLabelseu contém um índice de cluster baseado em 0 para a amostra armazenada no eut h linha da matriz de amostras.
# "pixel_vals" é a matriz de intrada, os dados para serem clusterisados
# "K" é o numero de clusters os quais seram divididos
# "None" Matriz de inteiros de entrada / saída que armazena os índices de cluster para cada amostra.
# "criteria" Os critérios de terminação do algoritmo, ou seja, o número máximo de iterações e / ou a precisão desejada.
# "10" Sinalizador para especificar o número de vezes que o algoritmo é executado usando diferentes rotulagens iniciais. O algoritmo retorna os rótulos que produzem a melhor compactação
# "cv2.KMEANS_RANDOM_CENTERS" Definir sinalizadores (apenas para evitar quebra de linha no código)
#-----
# "retval" É a soma da distância ao quadrado de cada ponto até seus centros correspondentes.
# "labels" Esta é a matriz de rótulos (igual a 'código' no artigo anterior) onde cada elemento marcado como '0', '1' .....
# "centers"  Esta é a matriz de centros de clusters.
centers = np.uint8(centers) #converte o valor para inteiro sem sinal de 8 bits ( 0 para 255).
segmented_data = centers[labels.flatten()]
segmented_image = segmented_data.reshape((b.shape))
pixel_labels = labels.reshape(img_lab.shape[0], img_lab.shape[1])
# displaying segmented image
plt.imshow(segmented_image)

#-------------

# Doing this, some unwanted pixels that are clustered in main cluster can be avoided.
# Ref - https://docs.opencv.org/3.4/d3/dc0/group__imgproc__shape.html#gac2718a64ade63475425558aa669a943a
pixel_labels = np.uint8(pixel_labels)
ret, components = cv2.connectedComponents(pixel_labels, connectivity=8)
plt.imshow(components, cmap='gray')

#---------
indices = []
for i in range(1, ret):
    row, col = np.where(components==i)
    indices.append(max(len(row), len(col)))
component = np.argmax(np.array(indices))
main_component = component+1   #indexing starts from 0, so we increment by 1 to get actual component index
# creating a mask and extracting pixels corresponding to cluster to which leaf belongs.
# 1 for actual leaf pixels and 0 for other pixels
mask = np.where(components==main_component, 1, 0)
B = image[:, :, 0]
G = image[:, :, 1]
R = image[:, :, 2]
# Extract only masked pixels
r = R*mask
g = G*mask
b = B*mask
final_img = np.dstack((r, g, b))
plt.imshow(final_img)

#---------
v_sum = np.sum(mask, axis=0)
h_sum = np.sum(mask, axis=1)
w = np.count_nonzero(v_sum)
h = np.count_nonzero(h_sum)
x_indices = np.where(v_sum != 0)
y_indices = np.where(h_sum != 0)
x = x_indices[0][0]
y = y_indices[0][0]
final_crop_img = final_img[y:y+h-1, x:x+w-1, :]
final_crop_img = np.uint8(final_crop_img)
img = cv2.resize(final_crop_img, (640,480), interpolation=cv2.INTER_AREA)
plt.imshow(img)
cv2.imwrite("img/saida/imagem-sem-fundo.jpg", final_crop_img)
plt.show()
