import os
import time
import numpy as np
import matplotlib.pyplot as plt

CSV_DIR = 'dataset/mnist_export'
NPY_DIR = 'dataset/npy'
SPLITS = ('train', 'val', 'test')


def load_split_csv(name):
    x = np.loadtxt(f'{CSV_DIR}/x_{name}.csv', delimiter=',', dtype=np.float32)
    y = np.loadtxt(f'{CSV_DIR}/y_{name}.csv', delimiter=',', dtype=np.int64)
    return x, y


def save_npy():
    os.makedirs(NPY_DIR, exist_ok=True)
    for name in SPLITS:
        x, y = load_split_csv(name)
        np.save(f'{NPY_DIR}/x_{name}.npy', x)
        np.save(f'{NPY_DIR}/y_{name}.npy', y)
        print(f'Saved {name}: x {x.shape}, y {y.shape}')


def compare_load_times():
    t0 = time.perf_counter()
    x_csv = np.loadtxt(f'{CSV_DIR}/x_train.csv', delimiter=',', dtype=np.float32)
    loadtxt_time = time.perf_counter() - t0

    t0 = time.perf_counter()
    x_npy = np.load(f'{NPY_DIR}/x_train.npy')
    load_time = time.perf_counter() - t0

    print(f'x_train np.loadtxt: {loadtxt_time} s')
    print(f'x_train np.load:    {load_time} s')


def load_split(name):
    if name not in SPLITS:
        raise ValueError(f'Invalid split: {name}')
    if not os.path.exists(f'{NPY_DIR}/x_{name}.npy'):
        save_npy()

    x = np.load(f'{NPY_DIR}/x_{name}.npy')
    y = np.load(f'{NPY_DIR}/y_{name}.npy')
    assert x.dtype == np.float32, f'x_{name} is {x.dtype}, expected float32'
    assert y.dtype == np.int64, f'y_{name} is {y.dtype}, expected int64'
    assert x.ndim == 2 and x.shape[1] == 784, f'x_{name} has shape {x.shape}, expected (N, 784)'
    assert y.ndim == 1, f'y_{name} has shape {y.shape}, expected (N,)'
    assert x.shape[0] == y.shape[0], f'x_{name} has {x.shape[0]} rows but y_{name} has {y.shape[0]}'
    return x, y


def inspect_split(x, y, name):
    print(f'x_{name}')
    print(x.shape)
    print(x.dtype)
    print(x.min())
    print(x.max())

    print(f'y_{name}')
    print(y.shape)
    print(y.dtype)
    print(y.min())
    print(y.max())
    print(y[:10])
    print(np.bincount(y))

    print(f'First image of x_{name}:')
    plt.imshow(x[0].reshape(28, 28), cmap="gray")
    plt.show()


if __name__ == '__main__':
    x_train, y_train = load_split('train')
    inspect_split(x_train, y_train, 'train')

    x_val, y_val = load_split('val')
    inspect_split(x_val, y_val, 'val')

    x_test, y_test = load_split('test')
    inspect_split(x_test, y_test, 'test')
