from keras.datasets import cifar10
import numpy as np
import matplotlib.pyplot as plt
class args:
    num_class = 10
    num_batch = 40
    batch_size = 125
    num_valid = 1000

def load_cifar10(args):
    num_perclass_train = args.num_batch * args.batch_size // 10
    num_perclass_valid = args.num_valid // 10
    num_class = args.num_class

    (x_train, y_train), (x_valid, y_valid) = cifar10.load_data()

    x_train, y_train = data_subset(x_train, y_train, num_perclass_train, num_class)
    x_valid, y_valid = data_subset(x_valid, y_valid, num_perclass_valid, num_class)
    return (x_train, y_train), (x_valid, y_valid)


def data_subset(x, y, num_perclass, num_class):
    n = x.shape[0] // num_class
    index_arr = np.argsort(np.squeeze(y))

    x_ret, y_ret = np.array(x[:num_perclass * num_class]), np.array(y[:num_perclass * num_class])

    for i in range(num_class):
        x_ret[i * num_perclass:(i + 1) * num_perclass] = x[index_arr[n * i : n * i + num_perclass]]
        y_ret[i * num_perclass:(i + 1) * num_perclass] = y[index_arr[n * i : n * i + num_perclass]]

    s = np.arange(n)
    np.random.shuffle(s)
    return x_ret[s], y_ret[s]

if __name__ == "__main__":
    a = args()
    x,y = load_cifar10(a)