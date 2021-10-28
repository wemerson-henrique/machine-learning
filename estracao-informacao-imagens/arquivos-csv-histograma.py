import csv
import cv2
import numpy as np
from matplotlib import pyplot as plt
import LeituraFormatacaoGravacaoDasImagens as Lfgi


f = open('dados_histograma.csv', 'w', newline='', encoding='utf-8')
w = csv.writer(f)
'''for i in range(5):
  w.writerow([i, i*2, i*3, "oi"])
w.close()'''

#----------------------------------
a = Lfgi.trabalhandoAsImagens()
listaImagens = a.lendoImagensDaPasta("img/37/")
nomeImagem = "Imagem"

for indice in range(len(listaImagens)):
  print(str(listaImagens[indice]))

  #--------------start here
  caminho = str(listaImagens[indice])

  img2 = cv2.imread(str(caminho))
  img = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
  hist = cv2.calcHist([img], [0], None, [256], [0, 256])
  '''arrayHist = hist
  dados = []
  maior = 0
  menor = 0
  cont = 0
  w.writerow(["Oorrencia:","Valor"])
  for indexi in range(len(hist)):
    cont=0
    for indexj in range(len(arrayHist)):
      if hist[indexi] == arrayHist[indexj]:
       cont+=1
    w.writerow([cont, indexi])'''
  cv2.imwrite("img/37Saida/"+nomeImagem+str(indice)+"-preto-e-branco.jpg",img)
  plt.savefig("img/37Saida/"+nomeImagem+str(indice)+"-preto-e-branco-Histograma.jpg")
  plt.close()


  img2 = cv2.imread(str(caminho), cv2.IMREAD_COLOR)
  color = ('b', 'g', 'r')
  for i, col in enumerate(color):
    histr = cv2.calcHist([img2], [i], None, [256], [0, 256])
    plt.plot(histr, color=col)
    plt.xlim(0, 256)

  cv2.imwrite("img/37Saida/" + nomeImagem + str(indice)+".jpg", img2)
  plt.savefig("img/37Saida/" + nomeImagem + str(indice) + "-colorida-Histograma.jpg")
  plt.close()

  image = img2
  img_lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
  L = img_lab[:, :, 0]
  a = img_lab[:, :, 1]
  b = img_lab[:, :, 2]
  fig, ax = plt.subplots(1, 3, figsize=(15, 15))
  ax[0].imshow(L)
  ax[1].imshow(a)
  ax[2].imshow(b)

  pixel_vals = b.flatten()
  pixel_vals = np.float32(pixel_vals)
  criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
  K = 2  # criação de clusters
  retval, labels, centers = cv2.kmeans(pixel_vals, K, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)

  centers = np.uint8(centers)
  segmented_data = centers[labels.flatten()]
  segmented_image = segmented_data.reshape((b.shape))
  pixel_labels = labels.reshape(img_lab.shape[0], img_lab.shape[1])
  plt.imshow(segmented_image)

  pixel_labels = np.uint8(pixel_labels)
  ret, components = cv2.connectedComponents(pixel_labels, connectivity=8)
  plt.imshow(components, cmap='gray')

  # ---------
  indices = []
  for i in range(1, ret):
    row, col = np.where(components == i)
    indices.append(max(len(row), len(col)))
  component = np.argmax(np.array(indices))
  main_component = component + 1

  mask = np.where(components == main_component, 1, 0)
  B = image[:, :, 0]
  G = image[:, :, 1]
  R = image[:, :, 2]

  r = R * mask
  g = G * mask
  b = B * mask
  final_img = np.dstack((r, g, b))
  plt.imshow(final_img)

  # ---------
  v_sum = np.sum(mask, axis=0)
  h_sum = np.sum(mask, axis=1)
  w = np.count_nonzero(v_sum)
  h = np.count_nonzero(h_sum)
  x_indices = np.where(v_sum != 0)
  y_indices = np.where(h_sum != 0)
  x = x_indices[0][0]
  y = y_indices[0][0]
  final_crop_img = final_img[y:y + h - 1, x:x + w - 1, :]
  final_crop_img = np.uint8(final_crop_img)
  img = cv2.resize(final_crop_img, (640, 480), interpolation=cv2.INTER_AREA)
  plt.imshow(img)
  plt.savefig("img/37Saida/" + nomeImagem + str(indice) + "-resultado.jpg")
  plt.close()

print("Tudo ocorreu bem")