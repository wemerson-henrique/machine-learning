import pandas as pd
from glob import glob
import cv2
#Lendo arquivos

# Vai ler todos os arquivos .jpg do caminho
arquivos = sorted(glob('img/mamao/*.jpg'))
print(str(arquivos[0]))
img1 = cv2.imread(arquivos[0])
cv2.imshow("Img1",img1)
cv2.waitKey(0)
#Lendo arquivos para um Dataframe

