# referencias: https://minerandodados.com.br/processamento-de-imagens-com-python/
import PIL

#bibliotecas para visualizar as imagens
from matplotlib import image
from numpy import asarray
from matplotlib import pyplot

from PIL import Image #importando classe image da biblioteca

'''image = Image.open("sigatoka.jpg") #vai carrega a imagem a ser trabalhada
print(image.format) #emprime o formato da imagem
print(image.mode) #modo de canal de pixels'''


data = image.imread("sigatoka.jpg") #carregando imagem como um array de pixels NumPy
#print(data)
'''print(data.dtype) #retorna o tipo de dado
print(data.shape) #mostra as carateristicas do arrey
print(data.max()) #valor maximo
print(data.min()) #valor minimo'''

'''pyplot.imshow(data) # exibe o array de pixels como uma imagem'''

image2 = Image.fromarray(data) #converte o array para imagem
'''image_cinza = image2.convert(mode="L") #coverte imagem para cinza
image_cinza.save("sagatoka_cinza.png",format="PNG") #comando par salvar'''

'''print(image2.size) #mostra tamanho da imgaem
image2.thumbnail((100,100)) #reduz tamanho da imagem proporcionalmente
print(image2.size) #mostra tamanho da imgaem'''

image = Image.open("sigatoka.jpg") #ira carregar a imagem na variavel "imagem"
image_resize = image.resize((200,200)) #ira redimencionar cem conciderar as proporções da imagem
print(image_resize.size)