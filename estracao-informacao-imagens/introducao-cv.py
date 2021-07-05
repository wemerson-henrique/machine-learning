
'''#referencias: https://professor.luzerna.ifc.edu.br/
ricardo-antonello/wp-content/uploads/sites/8/2017/02/
Livro-Introdu%C3%A7%C3%A3o-a-Vis%C3%A3o-Computacional-com-Python-e-OpenCV-3.pdf'''

# Importação das bibliotecas
import cv2
# Leitura da imagem com a função imread()
imagem = cv2.imread('sigatoka.jpg')

print('Largura em pixels: ', end='')
print(imagem.shape[1]) #largura da imagem
print('Altura em pixels: ', end='')
print(imagem.shape[0]) #altura da imagem
print('Qtde de canais: ', end='')
print(imagem.shape[2])
# Mostra a imagem com a função imshow
cv2.imshow("Nome da janela", imagem)
cv2.waitKey(0) #espera pressionar qualquer tecla
cv2.imwrite("saida.jpg", imagem) # Salvar a imagem no disco com função imwrite()
