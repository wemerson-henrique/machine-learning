import cv2
from glob import glob
import indentificacaoDaDoenca2 as IndDoen

arquivos = sorted(glob('img/mamao/*.jpg'))
print(str(arquivos[0]))
#img1 = IndDoen.Detecta_Doenca(img)
#cv2.imshow("Doença RGB", img.ConvercaoYCrCb())
#cv2.waitKey(0)