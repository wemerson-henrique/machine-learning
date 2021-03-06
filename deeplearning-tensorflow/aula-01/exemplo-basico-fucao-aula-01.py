#feito por wemerson em henrique em 22/06/2021
'''
o exeplo é para descobrir uma função
intradas: X= -1, 0, 0, 1, 2, 3, 4
saidas: -3, -1, 1, 3, 5, 7
função: y = 2 * x - 1

float hw_function(float x){
    float y = ( 2 * x ) - 1;
    return y;
}
'''

from tensorflow import *;
import numpy as np; #omenario abaixo

'''
NumPy é uma biblioteca Python usada largamente para operações com
listas/arrays, é muito utilisado matematica destacando suas funcionalidades com listas e arrays e é boa pois cria uma representação abstrata visual. Você pode pensar em um array NumPy como um tipo especial de lista com uma série de funções adicionais.
Para criar um array NumPy você deve especificar o tamanho e um método de inicialização. Veja um exemplo:
import numpy as np; #faz importação
nparray = np.zeros((5,5)); #diz quantas linhas e colunas
'''

def main():
    model = keras.Sequential([keras.layers.Dense(units=1, input_shape=[1])]); #comenario abaixo
    '''
    keras é uma biblioteca, python que roda em cima do tensorflow.
    Sequential é uma arquitetura de IA bem simples que tem apenas uma entrada e uma saida, "O modelo sequencial é útil quando temos apenas um entrada e uma saída."
    keras.layers.dense é dicações sobre a camada que nesse caso será apenas uma.
    units representa a quantidade de neurônios da camada.
    input_shape representa o formato dos dados de entrada, que necesse caso é um vetor com um dado.
    '''
    model.compile(optimizer='sgd', loss='mean_squared_error'); #comenario abaixo
    '''
    Essa filanha de comando vai fazer alguns difinições importante para o modelo.
    optimizer é definição da função (ou jeito) utilizado para fazer com que a rede aprenda e modifique os pesos dentro dela.
    loss é a função de erro utilisada, como é calculado o erro.
    '''
    x = np.array([-1.0, 0.0, 1.0, 2.0, 3.0, 4.0], dtype=float); #vetor entradass
    y = np.array([-3.0, -1.0, 1.0, 3.0, 5.0, 7.0], dtype = float); #vetor saidas
    model.fit( x, y, epochs = 500); #era mostra e os ciclos,eras de aprendisado
    print(model.predict([1.0])); #exemplo de teste pratico
if __name__ == '__main__':
    main();