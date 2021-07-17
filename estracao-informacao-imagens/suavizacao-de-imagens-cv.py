#-----------------------------------------------------7.1 Suavização por cálculo da média---------------------------------------------------------------------

'''No código abaixo percebemos que o método utilizado para a suavização pela média é
o método ‘blur’ da OpenCV. Os parâmetros são a imagem a ser suavizada e a janela de
suavização. Colocarmos números impars para gerar as caixas de cálculo pois dessa forma não
existe dúvida sobre onde estará o pixel central que terá seu valor atualizado. '''

'''from matplotlib import pyplot as plt
import numpy as np
import cv2
img = cv2.imread('img/entrada/sigatoka.jpg')
img = img[::2,::2] # Diminui a imagem
suave = np.vstack([
 np.hstack([img, cv2.blur(img, ( 3, 3))]),
 np.hstack([cv2.blur(img, (5,5)), cv2.blur(img, ( 7, 7))]),
 np.hstack([cv2.blur(img, (9,9)), cv2.blur(img, (11, 11))]),
 ])
cv2.imshow("Imagens suavisadas (Blur)", suave)
cv2.waitKey(0)'''

#-----------------------------------------------------77.2 Suavização pela Gaussiana---------------------------------------------------------------------

'''Ao invés do filtro de caixa é utilizado um kernel gaussiano. Isso é calculado através da
função cv2.GaussianBlur(). A função exige a especificação de uma largura e altura com
números impares e também, opcionalmente, é possível especificar a quantidade de desvios
padrão no eixo X e Y (horizontal e vertical). '''

'''import numpy as np
import cv2

img = cv2.imread('img/entrada/sigatoka.jpg')
img = img[::2,::2] # Diminui a imagem
suave = np.vstack([
    np.hstack([img,
        cv2.GaussianBlur(img, ( 3, 3), 0)]), np.hstack([cv2.GaussianBlur(img, ( 5, 5), 0),
        cv2.GaussianBlur(img, ( 7, 7), 0)]), np.hstack([cv2.GaussianBlur(img, ( 9, 9), 0),
        cv2.GaussianBlur(img, (11, 11), 0)]),
 ])
cv2.imshow("Imagem original e suavisadas pelo filtro Gaussiano", suave)

cv2.waitKey(0)'''

#-----------------------------------------------------7.3 Suavização pela mediana---------------------------------------------------------------------

'''Da mesma forma que os cálculos anteriores, aqui temos o cálculo de uma caixa ou
janela quadrada sobre um pixel central onde matematicamente se utiliza a mediana para
calcular o valor final do pixel. A mediana é semelhante à média, mas ela despreza os valores
muito altos ou muito baixos que podem distorcer o resultado. A mediana é o número que fica
examente no meio do intervalo.
A função utilizada é a cv2.medianBlur(img, 3) e o único argumento é o tamaho da
caixa ou janela usada.
É importante notar que este método não cria novas cores, como pode acontecer com os
ateriores, pois ele sempre altera a cor do pixel atual com um dos valores da vizinhança.
'''

'''import numpy as np
import cv2

img = cv2.imread('img/entrada/sigatoka.jpg')
img = img[::2,::2] # Diminui a imagem
suave = np.vstack([
 np.hstack([img,
 cv2.medianBlur(img, 3)]),
 np.hstack([cv2.medianBlur(img, 5),
 cv2.medianBlur(img, 7)]),
 np.hstack([cv2.medianBlur(img, 9),
 cv2.medianBlur(img, 11)]),
 ])
cv2.imshow("Imagem original e suavisadas pela mediana", suave)
cv2.waitKey(0)'''

#-----------------------------------------------------.4 Suavização com filtro bilateral---------------------------------------------------------------------

'''Este método é mais lento para calcular que os anteriores mas como vantagem
apresenta a preservação de bordas e garante que o ruído seja removido.
Para realizar essa tarefa, além de um filtro gaussiano do espaço ao redor do pixel 
também é utilizado outro cálculo com outro filtro gaussiano que leva em conta a diferença de
intensidade entre os pixels, dessa forma, como resultado temos uma maior manutenção das
bordas das imagem. A função usada é cv2.bilateralFilter() e o código usado segue abaixo:'''

'''import numpy as np
import cv2

img = cv2.imread('img/entrada/sigatoka.jpg')
img = img[::2,::2] # Diminui a imagem
suave = np.vstack([
 np.hstack([img,
 cv2.bilateralFilter(img, 3, 21, 21)]),
 np.hstack([cv2.bilateralFilter(img, 5, 35, 35),
 cv2.bilateralFilter(img, 7, 49, 49)]),
 np.hstack([cv2.bilateralFilter(img, 9, 63, 63),
 cv2.bilateralFilter(img, 11, 77, 77)])
 ])
cv2.imshow("Imagem original e suavisadas pela mediana", suave)
cv2.waitKey(0)'''