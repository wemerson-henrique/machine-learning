#Gravando arquivos CSV

import csv

# 1. cria o arquivo
f = open('dados_histograma.csv', 'w', newline='', encoding='utf-8')

# 2. cria o objeto de gravação
w = csv.writer(f)

# 3. grava as linhas
for i in range(5):
  w.writerow([i, i*2, i*3])

# Recomendado: feche o arquivo
w.close()

#Lendo arquivos CSV

import csv

# 1. abrir o arquivo
with open('0_dados_history.csv', encoding='utf-8') as arquivo_referencia:

  # 2. ler a tabela
  tabela = csv.reader(arquivo_referencia, delimiter=',')

  # 3. navegar pela tabela
  for l in tabela:
    id_autor = l[0]
    nome = l[1]

    print(id_autor, nome) # 191149, Diego C B Mariano