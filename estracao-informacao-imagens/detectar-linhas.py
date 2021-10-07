import cv2 as cv
import numpy as np
img = cv.imread(cv.samples.findFile('img/entrada/doenca-extraida-com-circulo.jpg'))
gray = cv.cvtColor(img,cv.COLOR_BGR2GRAY)
edges = cv.Canny(gray,50,150,apertureSize = 3)
comprimento_minimo = 1
intervalo_maximo = 200
lines = cv.HoughLinesP(image=edges, rho=1, theta=0.1 * np.pi/360, threshold=50, minLineLength=comprimento_minimo, maxLineGap=intervalo_maximo)
for line in lines:
    x1,y1,x2,y2 = line[0]
    cv.line(img,(x1,y1),(x2,y2),(0,255,0),2)
cv.imshow('Resultado',img)
cv.waitKey(0)