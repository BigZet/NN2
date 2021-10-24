import numpy as np
import struct
from functools import reduce
from scipy.optimize import minimize
import scipy
scipy.optimize.OptimizeResult
from matplotlib import pyplot as plt

# Вам предлагается обучить трехслойную нейронную сеть прямого распространения с сигмоидальными функциями активации на
# наборе данных MNIST - множестве изображений рукописных цифр и соответствующих метках 0-9.

# Ваша задача заполнить код в следующих функциях:
# logistic - вычисляет логистическую функцию от аргумента,
# compute_hypothesis - вычисляет выходы сети для заданного набора данных,
# compute_cost - вычисляет функцию стоимости,
# compute_cost_grad - вычисляет градиент функции стоимости с помощью алгоритма обратного распространения ошибки.

# По ходу работы смотрите в консоль. Там будут выводиться результаты написанных вами функций и ожидаемые результаты.
# Так вы сможете проверить себя.

# Ниже заданы "константы", с которыми можно "поиграть" - поварьировать их и посмотреть, что будет.
HIDDEN_LAYER_SIZE = 25  # Количество нейронов в скрытом слое.

# Всего у вас есть 60 000 изображений, поэтому сумма следующих двух констант не должна быть больше 60 000.
SAMPLES_TO_TRAIN = 1000  # Количество примеров в обучающей выборка, на ней будет производить подбор параметров модели.
SAMPLES_TO_TEST = 100  # Количество примеров в тестовой выборке, на ней будет оцениваться качество модели.


def one_hot_encode(y):

    # Функция кодирует вектор чисел с метками классов в матрицу, каждая строка которой является унитарным кодом
    # соответствующей метки.
    #y = np.array([int(i) for i in y])
    return np.array(np.arange(0., 10., 1) == y, dtype=np.int32)


def get_random_matrix(l_in, l_out):

    # Функция для генерации матрицы случайных параметров.

    return np.random.uniform(-0.5, 0.5, (l_out, l_in + 1))


def compute_cost_grad_approx(X, y, theta, lamb):

    # Фукнция для численного расчета градиента фукници стоимости. Такой метод полезно использовать при проверке
    # правильности аналитического расчета.

    eps = 1e-5
    grad = np.zeros(theta.size)
    perturb = np.zeros(theta.size)
    for i in range(theta.shape[0]):
        if i % 1000 == 0:
            print(f'{round(i / theta.shape[0] * 100, 2)}%')
        perturb[i] = eps
        grad[i] = (compute_cost(X, y, theta + perturb, lamb) - compute_cost(X, y, theta - perturb, lamb)) / (2 * eps)
        perturb[i] = 0
    return grad


def logistic(z):

    # Функция принимает аргумент z - скаляр, вектор или матрицу в виде объекта numpy.array()
    # Должна возвращать скяляр, вектор или матрицу (в зависимости от размерности z)
    # результатов вычисления логистической функции от элементов z

    result = 1 / (1 + np.exp(-z))

    # ВАШ КОД ЗДЕСЬ

    # =============

    return result


def logistic_grad(z):

    # Функция принимает аргумент z - скаляр, вектор или матрицу в виде объекта numpy.array()
    # Должна возвращать скяляр, вектор или матрицу (в зависимости от размерности z)
    # результатов вычисления производной логистической функции от элементов z

    #result = np.exp(-z) / ((1 + np.exp(-z)) ** 2)
    result = logistic(z) * (1-logistic(z))

    # ВАШ КОД ЗДЕСЬ

    # =============

    return result


def compute_hypothesis(X, theta):

    # Функция принимает матрицу данных X без фиктивного признака и вектор параметров theta.
    # Вектор theta является "разворотом" мартиц Theta1 и Theta2 в плоский вектор.
    # Функция должна возвращать матрицу выходов сети для каждого примера из обучающего набора,
    # а также матрицу входов нейронов скрытого слоя и матрицу выходов скрытого слоя (это понадобится для реализации
    # алгоритма обратного распространения ошибки).

    # Восстановление матриц Theta1 и Theta2 из ветора theta:
    Theta1 = theta[:HIDDEN_LAYER_SIZE * (X.shape[1] + 1)].reshape(HIDDEN_LAYER_SIZE, X.shape[1] + 1)
    Theta2 = theta[HIDDEN_LAYER_SIZE * (X.shape[1] + 1):].reshape(10, HIDDEN_LAYER_SIZE + 1)
    #print(Theta1.shape)
    #print(Theta2.shape)

    H = None  # Будущая матрица выходов, нужно заполнить
    Z2 = None  # Будущая матрица входов скрытого слоя, нужно заполнить
    A2 = None  # Будущая матрица выходов скрытого слоя, нужно заполнить

    # ВАШ КОД ЗДЕСЬ
    ones = np.ones(X.shape[0])
    X = np.column_stack((X, ones))
    Z2 = Theta1 @ X.T
    A2 = logistic(Z2)
    A2temp = np.column_stack((A2.T, ones)).T
    H_temp = Theta2 @ A2temp
    H = logistic(H_temp)
    #print(H.shape)

    # =============

    return H, Z2, A2




def compute_cost(X, Y, theta, lamb):

    # Функция принимает матрицу данных X без фиктивного признака, вектор верных классов y,
    # вектор параметров theta и параметр регуляризации lamb.
    # Вектор theta является "разворотом" мартиц Theta1 и Theta2 в плоский вектор.
    # Должна вернуть значение функции стоимости в точке theta.

    m, n = X.shape  # m - количество примеров в выборке, n - количество признаков у каждого примера
    cost = 0  # значение функции стоимости при заданных параметрах, его нужно посчитать

    # Восстановление матриц Theta1 и Theta2 из ветора theta:
    Theta1 = theta[:HIDDEN_LAYER_SIZE * (X.shape[1] + 1)].reshape(HIDDEN_LAYER_SIZE, X.shape[1] + 1)
    Theta2 = theta[HIDDEN_LAYER_SIZE * (X.shape[1] + 1):].reshape(10, HIDDEN_LAYER_SIZE + 1)

    # ВАШ КОД ЗДЕСЬ
    H, Z2, A2 = compute_hypothesis(X, theta)
    Y = np.array([one_hot_encode(i) for i in Y]).T
    sums_objects = []
    for i in range(Y.shape[1]):
        sums_neurons = []
        for j in range(Y.shape[0]):
            sums_neurons.append(Y[j, i] * np.log(H[j, i]) + (1 - Y[j, i]) * np.log(1 - H[j, i]))
        sums_objects.append(sum(sums_neurons))
    Q = -np.mean(sums_objects)
    R = (lamb/(2 * Y.shape[1]))*sum(theta)**2
    cost = Q + R
    # ==============

    return cost


def compute_cost_grad(X, Y, theta, lamb):

    # Функция принимает матрицу данных X без фиктивного признака, вектор верных классов y,
    # вектор параметров theta и параметр регуляризации lamb.
    # Вектор theta является "разворотом" мартиц Theta1 и Theta2 в плоский вектор.
    # Функция должна реализовать алгоритм обратного распространения ошибки и вернуть матрицы градиентов весов связей.

    m, n = X.shape  # m - количество примеров в выборке, n - количество признаков у каждого примера

    # Восстановление матриц Theta1 и Theta2 из ветора theta:
    Theta1 = theta[:HIDDEN_LAYER_SIZE * (X.shape[1] + 1)].reshape(HIDDEN_LAYER_SIZE, X.shape[1] + 1)
    Theta2 = theta[HIDDEN_LAYER_SIZE * (X.shape[1] + 1):].reshape(10, HIDDEN_LAYER_SIZE + 1)

    Theta1_grad = None  # Будущая матрица градиентов весов между входным и скрытым слоем, ее нужно заполнить
    Theta2_grad = None  # Будущая матрица градиентов весов между скрытым и выходным слоем, ее нужно заполнить

    # ВАШ КОД ЗДЕСЬ
    H, Z2, A2 = compute_hypothesis(X, theta)
    ones = np.ones(X.shape[0])
    X = np.column_stack((X, ones))
    A2 = np.column_stack((A2.T, ones)).T

    X = X.T
    Y = np.array([one_hot_encode(i) for i in Y]).T
    sigm_grad_2_arg = Theta2 @ A2
    sigm_grad_2 = logistic_grad(sigm_grad_2_arg)
    error_output = H - Y
    error_hidden = Theta2.T @ (error_output * sigm_grad_2)
    sigm_grad_1_arg = Theta1 @ X
    sigm_grad_1 = logistic_grad(sigm_grad_1_arg)
    grad2 = (error_output * sigm_grad_2) @ A2.T

    print(f"""
                Дополнительная справка:
                {H.shape=},
                {Z2.shape=},
                {A2.shape=},
                {Theta1.shape=},
                {Theta2.shape=},
                {X.shape=},
                {Y.shape=},
                {sigm_grad_2_arg.shape=}
                {sigm_grad_2.shape=}
                {error_output.shape=}
                {error_hidden.shape=}
                {sigm_grad_1_arg.shape=}
                {sigm_grad_1.shape=}
                {grad2.shape=}
                """)
    grad1 = (error_hidden[:-1, :] * sigm_grad_1) @ X.T
    Theta1_grad = grad1
    Theta2_grad = grad2
    #grad1 = (error_hidden[:, :] * Theta1 @ X) @ X.T






    #==============

    return np.concatenate((Theta1_grad, Theta2_grad), axis=None)


def load_data_idx(file_path):
    with open(file_path, 'rb') as input_file:
        magic = input_file.read(4)
        dims = int(magic[3])
        sizes = [struct.unpack('>L', input_file.read(4))[0] for _ in range(dims)]
        size = reduce(lambda x, y: x * y, sizes)
        data = np.array(list(input_file.read(size)), dtype=float)
        data = data.reshape(sizes)
        return data


# Загрузка данных
images = load_data_idx('images.idx')
features = images.reshape((images.shape[0], images.shape[1] * images.shape[2])) / 128 - 1.0

labels = load_data_idx('labels.idx')

X = features[:SAMPLES_TO_TRAIN]
Y = labels[:SAMPLES_TO_TRAIN]



#print(X.shape)
#print(Y.shape)
#print("Here")
#
X_test = features[SAMPLES_TO_TRAIN: SAMPLES_TO_TRAIN + SAMPLES_TO_TEST]
y_test = labels[SAMPLES_TO_TRAIN: SAMPLES_TO_TRAIN + SAMPLES_TO_TEST]

print(f'logistic(0) = {logistic(np.array(0))} (должно быть 0.5)\n'
      f'logistic(-10) = {logistic(np.array(-10))} (должно быть ~0)\n'
      f'logistic(10) = {logistic(np.array(10))} (должно быть ~1)')
print()
print(f'logistic_grad(0) = {logistic_grad(np.array(0))} (должно быть ~0.25)\n'
      f'logistic_grad(-10) = {logistic_grad(np.array(-10))} (должно быть ~0)\n'
      f'logistic_grad(10) = {logistic_grad(np.array(10))} (должно быть ~0)')

# im_size = int(features.shape[1] ** 0.5)
# hm = np.zeros((im_size * 10, im_size * 10))
# for i in range(100):
#     im_x = i % 10 * im_size
#     im_y = i // 10 * im_size
#     for j in range(features.shape[1]):
#         px_x = im_x + j % im_size
#         px_y = im_y + j // im_size
#         hm[10 * im_size - 1 - px_x, px_y] = images[i, j % im_size, j // im_size]
# plt.pcolor(hm, cmap='Greys')
# plt.show()
#
lamb = 0.5  # можно поварьировать параметр регуляризации и посмотреть, что будет

init_Theta1 = np.random.uniform(-0.12, 0.12, (HIDDEN_LAYER_SIZE, X.shape[1] + 1))
init_Theta2 = np.random.uniform(-0.12, 0.12, (10, HIDDEN_LAYER_SIZE + 1))

init_theta = np.concatenate((init_Theta1, init_Theta2), axis=None)

print(compute_hypothesis(X, init_theta)[0])

#compute_cost_grad(X, Y, init_theta, lamb)


# back_prop_grad = compute_cost_grad(X[:11], Y[:11], init_theta, 0)
# 
# print('Запуск численного расчета градиента (может занять какое-то время)')
# num_grad = compute_cost_grad_approx(X[:11], Y[:11], init_theta, 0)
# 
# print('Градиент, посчитанный аналитически: ')
# print(back_prop_grad)
# 
# print('Градиент, посчитанный численно (должен примерно совпадать с предыдущим): ')
# print(num_grad)
# 
# print()
# print('Относительное отклоение градиентов (должно быть < 1e-7): ')
# print((np.linalg.norm(back_prop_grad - num_grad) / np.linalg.norm(back_prop_grad + num_grad)))
# 
# print()
# print('Запуск оптимизации параметров сети (может занять какое-то время)')
# opt_theta_obj = minimize(lambda th: compute_cost(X, Y, th, lamb), init_theta,
#                           method='CG',
#                           jac=lambda th: compute_cost_grad(X, Y, th, lamb),
#                           options={'gtol': 1e-3, 'maxiter': 3000, 'disp': False})
#
# print('Минимизация функции стоимости ' + ('прошла успешно.' if opt_theta_obj.success else 'не удалась.'))
# print(opt_theta_obj)
#
# opt_theta = opt_theta_obj.x
# print('Функция стоимости при оптимальных параметрах (должна быть < 0.5): ')
# print(compute_cost(X, Y, opt_theta, lamb))
#
# hypos_train = compute_hypothesis(X, opt_theta)[0]
#
# true_preds_train_num = 0
# for k in range(hypos_train.shape[0]):
#     max_el = np.max(hypos_train[k, :])
#     true_preds_train_num += list(np.array(hypos_train[k, :] == max_el, dtype=np.int32)) == list(one_hot_encode(y[k]))
# print('Доля верно распознаных цифр в обучающем наборе:', true_preds_train_num / hypos_train.shape[0] * 100, '%')
#
# hypos_test = compute_hypothesis(X_test, opt_theta)[0]
#
# true_preds_test_num = 0
# for k in range(hypos_test.shape[0]):
#     max_el = np.max(hypos_test[k, :])
#     true_preds_test_num += list(np.array(hypos_test[k, :] == max_el, dtype=np.int32)) == list(one_hot_encode(y_test[k]))
# print('Доля верно распознаных цифр в тестовом наборе:', true_preds_test_num / hypos_test.shape[0] * 100, '%')
#
# if HIDDEN_LAYER_SIZE ** 0.5 % 1.0 == 0:
#     Theta1 = opt_theta[:HIDDEN_LAYER_SIZE * (X.shape[1] + 1)].reshape(HIDDEN_LAYER_SIZE, X.shape[1] + 1)
#     Theta1 = Theta1[:, 1:]
#     Theta1 = (Theta1 - Theta1.mean(axis=0)) / (Theta1.std(axis=0))
#     Theta1 = Theta1.reshape(HIDDEN_LAYER_SIZE, 28, 28)
#     im_size = int(features.shape[1] ** 0.5)
#     size_imgs = int(HIDDEN_LAYER_SIZE ** 0.5)
#     hm = np.zeros((im_size * size_imgs, im_size * size_imgs))
#     for i in range(HIDDEN_LAYER_SIZE):
#         im_x = i % size_imgs * im_size
#         im_y = i // size_imgs * im_size
#         for j in range(features.shape[1]):
#             px_x = im_x + j % im_size
#             px_y = im_y + j // im_size
#             hm[size_imgs * im_size - 1 - px_x, px_y] = Theta1[i, j % im_size, j // im_size]
#     plt.pcolor(hm, cmap='Greys')
#     plt.show()
