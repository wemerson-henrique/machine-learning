import cv2;

#img = cv2.imread('img/entrada/sigatoka1.jpeg');
img = cv2.imread('img/entrada/sigatoka.jpg');

img_branco_preto = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY);
img_bordas = cv2.Canny(img_branco_preto,100,200);


cv2.imshow("Img Original",img);
cv2.imshow("Img Preto e Branco", img_branco_preto);
cv2.imshow("Obordas detectadas", img_bordas);
cv2.waitKey(0)