import numpy as np
import cv2
def showImage(img):
    from matplotlib import pyplot as plt
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    plt.imshow(img)
    plt.show()
def main():
    img = cv2.imread("folha-de-mamao.jpg")
    print("As dimensões dessa imagem são: " + str(img.shape))

    altura, largura, cores = img.shape
    for y in range(0, altura):
        for x in range(0, largura):
            # posição x, y e a coordenada de cor (0 é azul)
            azul = img.item(y, x, 0)
            # posição x, y e a coordenada de cor (1 é verde)
            verde = img.item(y, x, 1)
            # posição x, y e a coordenada de cor (2 é vermelho)
            vermelho = img.item(y, x, 2)
            # na posição x, y e coordenada azul, atribui-se o valor 0
            img.itemset((y, x, 0), 0)
            # na posição x, y e coordenada vermelha, atribui-se o valor 0
            img.itemset((y, x, 1), 0)
            # como vamos manter o vermelho como está, não é necessário adicionar uma chamada para tal

    showImage(img)
main()