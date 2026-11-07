# Matrix Backprop MNIST

A from-scratch neural network for MNIST digit classification, written in NumPy. The forward pass, backprop and training all use plain matrix operations, with no deep learning framework.

## Status

Done so far:

- [x] Export MNIST from the pickle file to CSV
- [x] Cache the CSVs as `.npy` files for fast loading
- [x] Load each split with dtype and shape checks
- [x] Inspect the data (shapes, value ranges, label counts, first image)
- [x] Initialize weights (Xavier/Glorot normal)
- [ ] Forward pass
- [ ] Loss (softmax + cross-entropy)
- [ ] Backprop
- [ ] Training loop
- [ ] Evaluate on val/test

## Project layout

```
data.py                  Loads, caches and inspects the dataset splits
model.py                 Initializes the network parameters
main.py                  Entry point (empty for now)
utils/mnistConverter.py  One-off export of mnist.pkl to CSV
dataset/                 Data files (git-ignored)
  mnist.pkl
  mnist_export/          x_{split}.csv, y_{split}.csv
  npy/                   x_{split}.npy, y_{split}.npy (made by data.py)
```

## Data

The data is the classic `mnist.pkl` with three splits:

| Split | Images |
|-------|--------|
| train | 50,000 |
| val   | 10,000 |
| test  | 10,000 |

Each image is a row of 784 pixel values (28×28 flattened), stored as float32 in the range [0, 1]. Each label is a digit from 0 to 9.

### Pipeline

1. **`utils/mnistConverter.py`** unpickles `mnist.pkl` and writes one CSV per split for images (`x_*.csv`) and one for labels (`y_*.csv`). It writes to `resources/mnist_export` four folders above `utils/`, so the output then has to be copied into `dataset/mnist_export/`.
2. **`data.py`** does the rest:
   - `load_split_csv(name)` reads the CSVs with `np.loadtxt`, giving float32 images and int64 labels.
   - `save_npy()` converts every split to `.npy` in `dataset/npy/`.
   - `load_split(name)` loads a split from `.npy`. If the `.npy` files don't exist yet, it builds them first. It then checks:
     - `x` is float32 with shape `(N, 784)`
     - `y` is int64 with shape `(N,)`
     - `x` and `y` have the same number of rows
   - `compare_load_times()` times `np.loadtxt` on the CSV against `np.load` on the `.npy`. This is why the `.npy` cache exists.
   - `inspect_split(x, y, name)` prints shapes, dtypes, value ranges, the first 10 labels and the count of each digit, then shows the first image.

If you regenerate the CSVs, delete `dataset/npy/` first. Otherwise `load_split` keeps using the old `.npy` files.

## Model

`init_params(sizes, seed=42)` in `model.py` builds one `(W, b)` pair per layer:

- `W` has shape `(fan_in, fan_out)`. It is drawn from a standard normal distribution and scaled by `sqrt(2 / (fan_in + fan_out))` (Xavier/Glorot normal init).
- `b` starts at zero with shape `(fan_out,)`.
- Both are float32. The fixed seed makes runs reproducible.

The shapes are set up for row-vector inputs, so each layer computes `X @ W + b` with `X` shaped `(batch, fan_in)`.

The current architecture is `784 → 128 → 64 → 10`:

| Layer | W          | b      | Params  |
|-------|------------|--------|---------|
| 1     | (784, 128) | (128,) | 100,480 |
| 2     | (128, 64)  | (64,)  | 8,256   |
| 3     | (64, 10)   | (10,)  | 650     |
| Total |            |        | 109,386 |

## Running

Requires Python with `numpy` and `matplotlib`.

```bash
python data.py    # build the .npy cache if needed, then inspect each split
python model.py   # initialize the parameters and print their shapes and std values
```
