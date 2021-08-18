import cv2;
from matplotlib import pyplot as plt

img = cv2.imread('img/entrada/sigatoka3.jpeg'); #CONVERTE A IMAGEM EM VETOR
#img = cv2.imread('img/entrada/sigatoka.jpg');


img1 = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
h_eq = cv2.equalizeHist(img1)
#imgcinza = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY); #DEIRA PADÃRO DE CINZA

imgsuave = cv2.GaussianBlur(h_eq, (7,7), 0); #SUAVISA A IMAGEM
bordas =  cv2.Canny(imgsuave,100,200)

cv2.imshow("Imagem ", img)
cv2.imshow("Imagem melhorada", h_eq)
cv2.imshow("Imagem bordas", bordas)
cv2.waitKey(0)