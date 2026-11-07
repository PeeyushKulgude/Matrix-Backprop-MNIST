import numpy as np

def init_params(sizes, seed=42):
    rng = np.random.default_rng(seed)
    params = []
    for fan_in, fan_out in zip(sizes[:-1], sizes[1:]):
        std = np.sqrt(2 / (fan_in + fan_out))  # Xavier/Glorot normal
        W = (rng.standard_normal((fan_in, fan_out)) * std).astype(np.float32)
        b = np.zeros(fan_out, dtype=np.float32)
        params.append((W, b))
    return params


if __name__ == '__main__':
    params = init_params([784, 128, 64, 10])
    total_param_count = 0
    for i, (W, b) in enumerate(params, start=1):
        total_param_count += W.size + b.size
        expected_std = np.sqrt(2 / (W.shape[0] + W.shape[1]))
        print(f"layer {i}: W {W.shape} {W.dtype} actual_std={W.std():.8f} expected_std={expected_std:.8f} b {b.shape} {b.dtype}")
    print('Total params:', total_param_count)