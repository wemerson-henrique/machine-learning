import cv2
import numpy as np
#-------------Imagens de Teste-------------------------------------
#img = cv2.imread('img/16/40e9d306-b56d-4066-b081-895eb2cfed1f.jpg')
#img = cv2.imread('img/16/06b4eeba-8283-46a8-9582-4fc8fecec4c6.jpg')
#img = cv2.imread('img/37/2bbc7e78-bdea-42c9-aca7-3f37610c95d9.jpg')
#img = cv2.imread('img/entrada/folha-de-mamao-menor.jpg')
img = cv2.imread('img/entrada/sigatoka.jpg')
#img = cv2.imread('img/entrada/sigatoka1.jpeg')
#img = cv2.imread('img/entrada/sigatoka3.jpeg')
#img = cv2.imread('img/entrada/tomate1.jpg')
img1 = img

#-------------Convertendo Imagem e estraindo canais de cores-------------------------------------
img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
img_hsv = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2HSV)

H = img_hsv[:, :, 0]
S = img_hsv[:, :, 1]
G = img_rgb[:, :, 1]

#-------------Aplicando metodo de binarização-------------------------------------
img_hsv_gaussian = cv2.GaussianBlur (G, (5,5), 0)
ret3, otsu1 = cv2.threshold (img_hsv_gaussian, 0,255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
aplic_G = True

img = S
blur = cv2.GaussianBlur(img,(5,5),0)
# find normalized_histogram, and its cumulative distribution function
hist = cv2.calcHist([blur],[0],None,[256],[0,256])
hist_norm = hist.ravel()/hist.sum()
Q = hist_norm.cumsum()
bins = np.arange(256)
fn_min = np.inf
thresh = -1
for i in range(0,23):#1,256 valor original
    p1,p2 = np.hsplit(hist_norm,[i]) # probabilities
    q1,q2 = Q[i],Q[255]-Q[i] # cum sum of classes
    if q1 < 1.e-6 or q2 < 1.e-6:
        continue
    b1,b2 = np.hsplit(bins,[i]) # weights
    # finding means and variances
    m1,m2 = np.sum(p1*b1)/q1, np.sum(p2*b2)/q2
    v1,v2 = np.sum(((b1-m1)**2)*p1)/q1,np.sum(((b2-m2)**2)*p2)/q2
    # calculates the minimization function
    fn = v1*q1 + v2*q2
    if fn < fn_min:
        fn_min = fn
        thresh = i
# find otsu's threshold value with OpenCV function
ret, otsu = cv2.threshold(blur,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
print( "{} {}".format(thresh,ret) )
#aplic_G = False

#-------------Aplicando operações morfológicas-------------------------------------
if aplic_G == True:
    otsu1 = cv2.bitwise_not(otsu1)

con = cv2.bitwise_or(otsu1,otsu)
'''cv2.imshow("otsu1", otsu1)
cv2.imshow("otsu", otsu)
cv2.imshow("Juncao canal G e S", con)'''

kernel = np.ones((5,5),np.uint8)
dilation = cv2.dilate(con,kernel,iterations = 7)
erosion = cv2.erode(dilation,kernel,iterations = 8)
#fechamento = cv2.morphologyEx (otsu, cv2.MORPH_CLOSE, kernel)
#gradiente = cv2.morphologyEx (otsu, cv2.MORPH_GRADIENT, kernel)

#-------------Aplicando mascara-------------------------------------
image = img1
mask = np.zeros(image.shape[:2], dtype="uint8")
cv2.rectangle(mask, (0, 90), (290, 450), 255, -1) #"mask"= endica em qual imagem sera aplicado, (0, 90), (290, 450), 255, 5)
masked = cv2.bitwise_and(image, image, mask=erosion) #"image", image, mask=mask



#-------------Mostrando na tela-------------------------------------
cv2.imshow("H", H)
cv2.imshow("S", S)
cv2.imshow("G", G)
cv2.imshow("img_hsv", img_hsv)
cv2.imshow("otsu-banarização", otsu)
cv2.imshow("dilatação", dilation)
cv2.imshow("erosão", erosion)
cv2.imshow("Resultado", masked)
#cv2.imshow("fechamento", fechamento)
#cv2.imshow("gradiente", gradiente)
cv2.waitKey(0)