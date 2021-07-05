import cv2
imagem = cv2.imread('sigatoka.jpg')
(b, g, r) = imagem[0, 0] #veja que a ordem BGR e não RGB/ essa linha ira selecionar o pix de cordenada [0,0] e pegar o seu valor de cores
print('O pixel (0, 0) tem as seguintes cores:') #mostrara uma mensagem
print('Vermelho:', r, 'Verde:', g, 'Azul:', b) #ira exibir o padão de cores do pix que foi celecionado

#---------------------------------------------------------------------------------------------------------------------------

'''import cv2
imagem = cv2.imread('sigatoka.jpg') #seleciona a imagem

# os for iram percorre a igagem identificando cada pixel
for y in range(0, imagem.shape[0]):
    for x in range(0, imagem.shape[1]):
        imagem[y, x] = (255,0,0) #ira mudar o padrão RGB de cada pixel de modo que a imagem fique toda azul
cv2.imshow("Imagem modificada", imagem) #abre uma tela coma a imagem
cv2.waitKey(0) #espera pressionar qualquer tecla'''

#---------------------------------------------------------------------------------------------------------------------------

'''import cv2
imagem = cv2.imread('sigatoka.jpg')
for y in range(0, imagem.shape[0]): #percorre linhas
    for x in range(0, imagem.shape[1]): #percorre colunas
        imagem[y, x] = (x%256,y%256,x%256) #ira mudar o padrão RGB de cada pixel de modo que cada pixel rebera o valor do resto da divisão por 256
cv2.imshow("Imagem modificada", imagem) #abre uma tela coma a imagem
cv2.waitKey(0) #espera pressionar qualquer tecla'''

#---------------------------------------------------------------------------------------------------------------------------

'''import cv2
imagem = cv2.imread('sigatoka.jpg')
for y in range(0, imagem.shape[0], 1): #percorre as linhas
    for x in range(0, imagem.shape[1], 1): #percorre as colunas
        imagem[y, x] = (0,(x*y)%256,0)
cv2.imshow("Imagem modificada", imagem)
cv2.waitKey(0)'''

#---------------------------------------------------------------------------------------------------------------------------

'''Com mais uma pequena modificação temos o código abaixo. O objetivo agora é saltar
a cada 10 pixels ao percorrer as linhas e mais 10 pixels ao percorrer as colunas. A cada salto é
criado um quadrado amarelo de 5x5 pixels. Desta vez parte da imagem orignal é preservada e
podemos ainda observar a ponte por baixo da grade de quadrados amarelos.'''

'''import cv2
imagem = cv2.imread('sigatoka.jpg')
for y in range(0, imagem.shape[0], 10): #percorre linhas
    for x in range(0, imagem.shape[1], 10): #percorre colunas
        imagem[y:y+5, x: x+5] = (0,255,255)
cv2.imshow("Imagem modificada", imagem)
cv2.waitKey(0)'''

#---------------------------------------------------------------------------------------------------------------------------
