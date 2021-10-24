import numpy as np
import time
import struct
from functools import reduce
from scipy.optimize import minimize
from matplotlib import pyplot as plt

def load_data_idx(file_path):
    with open(file_path, 'rb') as input_file:
        magic = input_file.read(4)
        dims = int(magic[3])
        sizes = [struct.unpack('>L', input_file.read(4))[0] for _ in range(dims)]
        size = reduce(lambda x, y: x * y, sizes)
        data = np.array(list(input_file.read(size)), dtype=float)
        data = data.reshape(sizes)
        return data


def one_hot_encode(Y: np.ndarray):
    """
    Функция перекодировки значений (0, 10)
    :param Y: Исходные данные преставлены вектором, где y_i принадлежит интервалу целых чисел [0, 10]
    :return: Данные в виде 1 = (0, 1, 0, 0, 0, 0, 0, 0, 0, 0)
    """
    return np.array(np.arange(0, 10, 1) == Y, dtype=np.int32)


class NN:
    def __init__(self, hidden_layer_size=25, samples_to_train=1000, samples_to_test=100):
        self.H = hidden_layer_size
        self.samples_to_train = samples_to_train
        self.samples_to_test = samples_to_test
        #self.temp_vars = {} Не факт что пригодится, перед сдачей убери

    def fit(self, X: np.ndarray, Y: np.ndarray, lamb = 1):
        self.Q = 0
        self.lastQ = 1
        """
        Функция для обучения модели по данным
        :param X: Исходный dataframe без фиктивного признака содержит обучаемую и тестируемую выборки
        :param Y: Ответы, заранее приведены к виду 1 = (0, 1, 0, 0, 0, 0, 0, 0, 0, 0)
        :param lamb: Параметр регуляризации
        :return: None
        """
        self.X = X
        self.Y = Y
        self.lamb = lamb
        self.size = X.shape[1]
        #self.X = np.vstack((np.ones(self.size), self.X))

        self.N = self.X.shape[0]
        self.M = self.Y.shape[0]


        # Изначально X принимает вид:
        #
        # | Номер объекта | Признак 1 | Признак 2 | ... | Признак N |
        # |---------------|-----------|-----------|-----|-----------|
        # | 1             |           |           |     |           |
        # | 2             |           |           |     |           |
        # | ...           |           |           |     |           |
        # | self.size     |           |           |     |           |
        #
        # Но для удобства работы и X и Y должны быть транспонированы, т.е. иметь вид:
        #
        # | Номер признака | Объект 1 | Объект 2 | ... | self.size |
        # |----------------|----------|----------|-----|-----------|
        # | Признак 1      |          |          |     |           |
        # | Признак 2      |          |          |     |           |
        # | ...            |          |          |     |           |
        # | Признак N      |          |          |     |           |

        assert X.shape[1] > X.shape[0], 'X на входе не транспонирован'
        assert Y.shape[1] > Y.shape[0], 'Y на входе не транспонирован'
        assert self.size == self.Y.shape[1], 'Входные и выходные данные различаются по количеству объектов в выборке'
        assert self.size >= (
                self.samples_to_test + self.samples_to_train), f'Не хватает данных для обучающей: {self.samples_to_train} и тестирующей {self.samples_to_test} выборок'



        self.T = X[:self.samples_to_test]
        self.X = X[-self.samples_to_train:]
        self.Z = []

        self.w1 = self._random_weight(neurons_out=self.H, neurons_in=self.N)
        self.w2 = self._random_weight(neurons_out=self.M, neurons_in=self.H)
        self._print()

        while abs(self.Q - self.lastQ)>0.005 and self.Q != np.NaN:
            self.lastQ = self.Q
            self._step()
            print(f"Q: {self.Q}")

    def _print(self):
        print(
            f"""
Начало обучения. Стартовые данные:
    {self.X.shape=},
    {self.Y.shape=},
    {self.N=},
    {self.M=},
    {self.size=},
    {self.H=},
    {self.samples_to_test=},
    {self.samples_to_train=}
    {self.w1.shape=} // с учетом фиктивного
    {self.w2.shape=} // с учетом фиктивного"""
        )

    def _step(self):
        self._forwardProp()
        self._compute_cost()
        self._backProp()
        self.w1 = self.w1 - self.grad1
        self.w2 = self.w2 - self.grad2

    def _forwardProp(self):
        print("\nForwardProp: ")
        print("\tПодсчет U")
        self.U = self._compute_layer(self.X, self._logistic, self.w1)
        print(f"\t\t{self.U.shape=}")
        print("\tПодсчет A")
        self.A = self._compute_layer(self.U, self._logistic, self.w2)
        print(f"\t\t{self.A.shape=}")


    def _compute_layer(self, X:np.ndarray, sigma, weights)->np.ndarray:
        #add fictive x0
        X = np.vstack((np.ones(X.shape[1]), X))
        print(f"\t\tВход внутри преобразуется к {X.shape=}")
        X_weighted = weights @ X
        self.Z.append(X_weighted) #Количество входов на скрытый слой, т.е. до применения сигмоиды
        X_sigmoided = sigma(X_weighted)
        return X_sigmoided

    def _random_weight(self, neurons_in, neurons_out):
        return np.random.uniform(-0.5, 0.5, (neurons_out, neurons_in + 1)).reshape((neurons_out, neurons_in+1))

    def _logistic(self, x):
        return 1 / (1 + np.exp(-x))

    def _logistic_grad(self, x):
        return np.exp(-x) / ((1 + np.exp(-x)) ** 2)


    def _compute_cost(self):
        A = self.A
        Y = self.Y
        weights_1 = self.w1.copy()
        weights_2 = self.w2.copy()
        lamb = self.lamb
        print("\nCompute cost:")
        assert A.shape == Y.shape, 'Размерности ответов (Y) и выхода нейросети (A) не равны'

        sums_objects = []
        for i in range(Y.shape[1]):
            sums_neurons = []
            for j in range(Y.shape[0]):
                sums_neurons.append(Y[j, i] * np.log(A[j, i]) + (1 - Y[j, i]) * np.log(1 - A[j, i]))
            print(f"\t\tНомер объекта #{i}. Функция потерь вычисленная на выходе:{sums_neurons}.")
            sums_objects.append(sum(sums_neurons))
        print(f"\tСумма значений на выходе для каждого объекта:{sums_objects}")
        Q = -np.mean(sums_objects)
        print(f"\tЗначение функции стоимости без регуляризации {Q}")
        
        flat_w1 = weights_1
        flat_w1.shape = flat_w1.size,
        flat_w2 = weights_2
        flat_w2.shape = flat_w2.size,

        R = (-lamb/(2 * Y.shape[1]))*(sum(flat_w1) + sum(flat_w2))
        print(f"\tВычитаемый 'регулятор': {R=}")

        Q_reg = Q + R
        print(f"\tЗначение функции стоимости c регуляризацией {Q_reg}")
        self.Q = Q_reg

    def _backProp(self):
        print("\nBackProp: ")
        X = self.X
        Y = self.Y
        A = self.A
        U = self.U
        Z2 = self.Z[0]
        Z3 = self.Z[1]
        weights_1 = self.w1.copy()
        weights_2 = self.w2.copy()

        self.error_output = A - Y
        print(f"\tРазмерность ошибки выходного слоя должна быть ({self.M}x{A.shape[1]}) {self.error_output.shape=}")

        print(f"""
        Дополнительная справка:
        {X.shape=},
        {Y.shape=},
        {self.error_output.shape=},
        {A.shape=},
        {U.shape=},
        {Z2.shape=},
        {Z3.shape=},
        {weights_1.shape=},
        {weights_2.shape=}
        """)

        X = np.vstack((np.ones(X.shape[1]), X))
        print(f"\tX внутри преобразуется к {X.shape=}")
        U = np.vstack((np.ones(U.shape[1]), U))
        print(f"\tU внутри преобразуется к {U.shape=}")
        Z2 = np.vstack((np.ones(Z2.shape[1]), Z2))
        print(f"\tZ2 внутри преобразуется к {Z2.shape=}")
        Z3 = np.vstack((np.ones(Z3.shape[1]), Z3))
        print(f"\tZ3 внутри преобразуется к {Z3.shape=}")
##Очень странный прием
        weights_1 = np.vstack((weights_1, (np.zeros(weights_1.shape[1]))))
        print(f"\tWeights 1 внутри преобразуется к {weights_1.shape=}")

        sigm_grad_2_arg = weights_2 @ U
        print(f"\tРазмерность аргумента производной сигмоиды между 2 и 3 слоем: {sigm_grad_2_arg.shape=}")

        self.sigm_grad_2 = self._logistic_grad(sigm_grad_2_arg)
        print(f"\tРазмерность производной сигмоиды между 2 и 3 слоем: {self.sigm_grad_2.shape=}")
        print(f"\tЧасть производной сигмоиды между 2 и 3 слоем: {self.sigm_grad_2[1, :5]}")

        self.error_hidden = weights_2.T @ (self.error_output * self.sigm_grad_2)
        print(f"\tРазмерность ошибки скрытого слоя должна быть ({self.H+1}x{U.shape[1]}) {self.error_hidden.shape=}")

        sigm_grad_1_arg = weights_1 @ X
        print(f"\tРазмерность аргумента производной сигмоиды между 1 и 2 слоем: {sigm_grad_1_arg.shape=}")

        self.sigm_grad_1 = self._logistic_grad(sigm_grad_1_arg)
        print(f"\tРазмерность производной сигмоиды между 1 и 2 слоем: {self.sigm_grad_1.shape=}")
        print(f"\tЧасть производной сигмоиды между 1 и 2 слоем: {self.sigm_grad_1[1, :5]}")

        self.grad2 = (self.error_output * self.sigm_grad_2) @ U.T
        print(f"\tРазмерность частной производной по весам между 2 и 3 слоем: {self.grad2.shape=}")


##Продолжение странного приема
        self.grad1 = ((self.error_hidden * self.sigm_grad_1) @ X.T)[1:, :]
        print(f"\tРазмерность частной производной по весам между 1 и 2 слоем: {self.grad1.shape=}")



def main():
    # X = np.random.rand(10 * 50)
    # X = X.reshape((10, 50))
    # Y = np.random.rand(2 * 50)
    # Y = Y.reshape((2, 50))

    images = load_data_idx('images.idx')
    features = images.reshape((images.shape[0], images.shape[1] * images.shape[2])) / 128 - 1.0

    labels = load_data_idx('labels.idx')

    X = features
    Y = labels

    temp = NN(hidden_layer_size=100, samples_to_train=1000, samples_to_test=0)
    temp.fit(X.T, one_hot_encode(Y).T)


if __name__ == '__main__':
    main()
