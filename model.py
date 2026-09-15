"""
Build a Trainable CNN from Scratch in NumPy

Assembled from your step-by-step solutions.
"""

import numpy as np

# Step 1 - argmax_rows
def argmax_rows(matrix):
    # TODO: return the index of the largest element in each row of a 2D array

    return np.argmax(matrix, axis = 1)

# Step 2 - row_max
import numpy as np

def row_max(matrix):
    # TODO: return the maximum value of each row of `matrix` with keepdims True for broadcasting.
    return np.max(matrix, axis = 1, keepdims = True)

# Step 3 - row_sum
import numpy as np

def row_sum(matrix):
    """Return per-row sums of a 2D array with shape (N, 1)."""
    # TODO: return the sum along axis 1 keeping the reduced dimension
    return np.sum(matrix, axis = 1, keepdims = True)

# Step 4 - exp_shifted
import numpy as np

def exp_shifted(logits):
    """Subtract per-row max from logits and exponentiate elementwise."""
    # TODO: shift each row of logits by its max and return elementwise exp
    max_row = row_max(logits)

    return np.exp(logits - max_row)

# Step 5 - stable_softmax
def stable_softmax(logits):
    # TODO: Compute a numerically stable softmax row-wise over (N, C) logits.
    max_row = row_max(logits)

    shifted = exp_shifted(logits)
    exp_sum = row_sum(shifted)

    return shifted / exp_sum

# Step 6 - one_hot
def one_hot(labels, num_classes):
    # TODO: convert integer labels into a (N, num_classes) one-hot float matrix
    
    return np.eye(num_classes)[labels]

# Step 7 - gather_true_class_probs
def gather_true_class_probs(probs, labels):
    # TODO: return probs[i, labels[i]] for every row i as a 1D length-N array.
    return probs[np.arange(len(labels)), labels]

# Step 8 - cross_entropy_loss
import numpy as np

def cross_entropy_loss(probs, labels, eps=1e-12):
    true_probs = gather_true_class_probs(probs, labels)

    return float(-np.mean(np.log(true_probs + eps)))

# Step 9 - accuracy
def accuracy(logits_or_probs, labels):
    # TODO: return the fraction of rows whose argmax matches the integer label.
    max_probs = argmax_rows(logits_or_probs)

    return np.sum(max_probs == labels) / len(labels)

# Step 10 - he_std
def he_std(fan_in):
    # TODO: return the He initialization standard deviation sqrt(2 / fan_in).
    return np.sqrt(2 / fan_in)

# Step 11 - he_init
def he_init(shape, fan_in, seed):
    # TODO: sample a weight tensor from a normal distribution scaled by He std using the seed.
    np.random.seed(seed)
    return np.random.normal(0.0, he_std(fan_in), size=shape)

# Step 12 - init_zero_bias
import numpy as np

def init_zero_bias(length):
    # TODO: return a 1D float array of zeros with the given length.
    return np.zeros(length)

# Step 13 - pad_2d
def pad_2d(images, pad):
    # TODO: zero-pad the spatial (H, W) dims of a 4D (N, C, H, W) tensor by `pad` on each side.
    if pad == 0:
        return images

    # - Dim 0 (N): không đệm (0, 0)
    # - Dim 1 (C): không đệm (0, 0)
    # - Dim 2 (H): đệm pad giá trị trên và dưới (pad, pad)
    # - Dim 3 (W): đệm pad giá trị trái và phải (pad, pad)
    pad_width = ((0, 0), (0, 0), (pad, pad), (pad, pad))

    return np.pad(images, pad_width, mode="constant", constant_values=0)

# Step 14 - output_spatial_size
def output_spatial_size(input_size, kernel, stride, padding):
    # TODO: return the conv/pool output spatial dimension from input_size, kernel, stride, padding
    
    return (input_size + 2 * padding - kernel) // stride + 1

# Step 15 - im2col
import numpy as np


def im2col(
    images: np.ndarray,
    kernel_h: int,
    kernel_w: int,
    stride: int,
    padding: int,
) -> np.ndarray:
    """Vectorized im2col (0 loop) matching the test suite format: each patch is a row.

    Reuses output_spatial_size and pad_2d.
    """
    N, C, H, W = images.shape

    out_h = output_spatial_size(H, kernel_h, stride, padding)
    out_w = output_spatial_size(W, kernel_w, stride, padding)
    images_padded = np.ascontiguousarray(pad_2d(images, padding))

    sN, sC, sH, sW = images_padded.strides

    shape = (N, out_h, out_w, C, kernel_h, kernel_w)
    strides = (sN, stride * sH, stride * sW, sC, sH, sW)

    patches = np.lib.stride_tricks.as_strided(
        images_padded, shape=shape, strides=strides
    )

    num_patches = N * out_h * out_w
    patch_dim = C * kernel_h * kernel_w

    return np.ascontiguousarray(patches).reshape(num_patches, patch_dim)

# Step 16 - col2im
import numpy as np


def col2im(
    cols: np.ndarray,
    input_shape: tuple,
    kernel_h: int,
    kernel_w: int,
    stride: int,
    padding: int,
) -> np.ndarray:
    """Re-roll a (N*out_h*out_w, C*kh*kw) column matrix back into a (N, C, H, W) tensor.

    Reuses output_spatial_size.
    """
    N, C, H, W = input_shape

    out_h = output_spatial_size(H, kernel_h, stride, padding)
    out_w = output_spatial_size(W, kernel_w, stride, padding)

    cols_reshaped = cols.reshape(N, out_h, out_w, C, kernel_h, kernel_w)

    H_pad = H + 2 * padding
    W_pad = W + 2 * padding
    images_padded = np.zeros((N, C, H_pad, W_pad), dtype=cols.dtype)

    for kh in range(kernel_h):
        h_end = kh + out_h * stride
        for kw in range(kernel_w):
            w_end = kw + out_w * stride

            patch_slice = cols_reshaped[:, :, :, :, kh, kw].transpose(0, 3, 1, 2)

            images_padded[:, :, kh:h_end:stride, kw:w_end:stride] += patch_slice

    return images_padded[:, :, padding : padding + H, padding : padding + W]

# Step 17 - conv2d_forward
import numpy as np


def conv2d_forward(
    x: np.ndarray,
    weights: np.ndarray,
    bias: np.ndarray,
    stride: int,
    padding: int,
) -> tuple[np.ndarray, tuple]:
    """Forward pass for a 2D convolution layer using im2col (row-patch convention).

    Args:
        x: Input tensor of shape (N, C_in, H, W).
        weights: Filter weights of shape (C_out, C_in, kernel_h, kernel_w).
        bias: Bias vector of shape (C_out,).
        stride: Stride step size.
        padding: Zero-padding size on each spatial side.

    Returns:
        Tuple of (out, cache) where:
            - out: Output tensor of shape (N, C_out, out_h, out_w).
            - cache: Tuple containing (x, weights, bias, stride, padding, x_cols)
            for backward pass.
    """
    N, C_in, H, W = x.shape
    C_out, _, kernel_h, kernel_w = weights.shape

    out_h = output_spatial_size(H, kernel_h, stride, padding)
    out_w = output_spatial_size(W, kernel_w, stride, padding)

    x_cols = im2col(x, kernel_h, kernel_w, stride, padding)

    w_row = weights.reshape(C_out, -1)

    out = x_cols @ w_row.T
    if bias is not None:
        out += bias

    out = out.reshape(N, out_h, out_w, C_out).transpose(0, 3, 1, 2)

    cache = {
        "x": x,
        "weights": weights,
        "bias": bias,
        "stride": stride,
        "padding": padding,
        "kernel_h": kernel_h,
        "kernel_w": kernel_w,
        "cols": x_cols,
    }

    return out, cache

# Step 18 - conv2d_grad_input
def conv2d_grad_input(d_out: np.ndarray, cache: dict) -> np.ndarray:
    """Backpropagate output gradients through convolution to compute gradient w.r.t input (dx).

    Reuses col2im.

    Args:
        d_out: Upstream gradient tensor of shape (N, C_out, out_h, out_w).
        cache: Dictionary from conv2d_forward containing: 'x', 'weights',
            'stride', 'padding', 'kernel_h', 'kernel_w'.

    Returns:
        dx: Gradient tensor w.r.t input x, shape (N, C_in, H, W).
    """
    x = cache["x"]
    weights = cache["weights"]
    stride = cache["stride"]
    padding = cache["padding"]
    kernel_h = cache["kernel_h"]
    kernel_w = cache["kernel_w"]

    C_out = weights.shape[0]
    d_out_flat = d_out.transpose(0, 2, 3, 1).reshape(-1, C_out)

    w_row = weights.reshape(C_out, -1)

    d_cols = d_out_flat @ w_row

    dx = col2im(d_cols, x.shape, kernel_h, kernel_w, stride, padding)

    return dx

# Step 19 - conv2d_grad_weights
def conv2d_grad_weights(d_out: np.ndarray, cache: dict) -> np.ndarray:
    """Compute gradient of loss w.r.t weights (dW) using im2col cache.

    Args:
        d_out: Upstream gradient tensor of shape (N, C_out, out_h, out_w).
        cache: Dictionary from conv2d_forward containing: 'weights', 'cols' (or
            'x', 'stride', 'padding', 'kernel_h', 'kernel_w').

    Returns:
        dW: Gradient tensor w.r.t weights, shape (C_out, C_in, kernel_h,
        kernel_w).
    """
    weights = cache["weights"]
    C_out = weights.shape[0]

    if "cols" in cache:
        cols = cache["cols"]
    else:
        cols = im2col(
            cache["x"],
            cache["kernel_h"],
            cache["kernel_w"],
            cache["stride"],
            cache["padding"],
        )

    d_out_flat = d_out.transpose(0, 2, 3, 1).reshape(-1, C_out)

    dW_row = d_out_flat.T @ cols

    return dW_row.reshape(weights.shape)

# Step 20 - conv2d_grad_bias
def conv2d_grad_bias(d_out: np.ndarray) -> np.ndarray:
    """Return a length C_out gradient by reducing d_out over batch and spatial axes."""
    return np.sum(d_out, axis=(0, 2, 3))

# Step 21 - conv2d_backward
def conv2d_backward(
    d_out: np.ndarray, cache: dict
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return (dx, dW, db) using the conv2d gradient helpers and the forward cache."""
    dx = conv2d_grad_input(d_out, cache)
    dW = conv2d_grad_weights(d_out, cache)
    db = conv2d_grad_bias(d_out)
    return dx, dW, db

# Step 22 - maxpool2d_forward
def maxpool2d_forward(
    x: np.ndarray, kernel: int, stride: int
) -> tuple[np.ndarray, dict]:
    """Run 2D max pooling over a 4D tensor (N, C, H, W) with no padding.

    Args:
        x: Input tensor of shape (N, C, H, W).
        kernel: Spatial size of the square pooling window.
        stride: Stride step size between windows.

    Returns:
        out: Max-pooled tensor of shape (N, C, out_h, out_w).
        cache: Dict containing 'x_shape', 'argmax', 'kernel', and 'stride'.
    """
    N, C, H, W = x.shape

    out_h = output_spatial_size(H, kernel, stride, 0)
    out_w = output_spatial_size(W, kernel, stride, 0)

    out = np.zeros((N, C, out_h, out_w), dtype=x.dtype)
    argmax = np.zeros((N, C, out_h, out_w), dtype=int)

    for i in range(out_h):
        h_start = i * stride
        h_end = h_start + kernel
        for j in range(out_w):
            w_start = j * stride
            w_end = w_start + kernel

            window = x[:, :, h_start:h_end, w_start:w_end]
            window_flat = window.reshape(N, C, -1)

            out[:, :, i, j] = np.max(window_flat, axis=-1)
            argmax[:, :, i, j] = np.argmax(window_flat, axis=-1)

    cache = {
        "x_shape": x.shape,
        "argmax": argmax,
        "kernel": kernel,
        "stride": stride,
    }

    return out, cache

# Step 23 - scatter_grad_window
def scatter_grad_window(
    grad_value: float, argmax_index: int, kernel: int
) -> np.ndarray:
    """Place grad_value at the argmax position within a (kernel, kernel) zero array."""
    window = np.zeros((kernel, kernel), dtype=float)
    window.flat[int(argmax_index)] = grad_value
    return window

# Step 24 - maxpool2d_backward
def maxpool2d_backward(d_out: np.ndarray, cache: dict) -> np.ndarray:
    """Scatter each d_out value to the cached argmax position in its window."""
    x = cache["x"]
    kernel = cache["kernel"]
    stride = cache["stride"]
    argmax = cache["argmax"]

    N, C, H, W = x.shape
    out_h, out_w = d_out.shape[2], d_out.shape[3]
    dx = np.zeros_like(x, dtype=float)

    for n in range(N):
        for c in range(C):
            for i in range(out_h):
                h_start = i * stride
                h_end = h_start + kernel
                for j in range(out_w):
                    w_start = j * stride
                    w_end = w_start + kernel

                    idx = argmax[n, c, i, j]
                    val = d_out[n, c, i, j]

                    dx[n, c, h_start:h_end, w_start:w_end] += scatter_grad_window(
                        val, idx, kernel
                    )

    return dx

# Step 25 - relu_forward (not yet solved)
# TODO: implement

# Step 26 - relu_backward (not yet solved)
# TODO: implement

# Step 27 - flatten_forward (not yet solved)
# TODO: implement

# Step 28 - flatten_backward (not yet solved)
# TODO: implement

# Step 29 - linear_forward (not yet solved)
# TODO: implement

# Step 30 - linear_grad_input (not yet solved)
# TODO: implement

# Step 31 - linear_grad_weights (not yet solved)
# TODO: implement

# Step 32 - linear_grad_bias (not yet solved)
# TODO: implement

# Step 33 - linear_backward (not yet solved)
# TODO: implement

# Step 34 - softmax_cross_entropy_forward (not yet solved)
# TODO: implement

# Step 35 - softmax_cross_entropy_backward (not yet solved)
# TODO: implement

# Step 36 - sgd_step (not yet solved)
# TODO: implement

# Step 37 - adam_update_m (not yet solved)
# TODO: implement

# Step 38 - adam_update_v (not yet solved)
# TODO: implement

# Step 39 - adam_bias_correct (not yet solved)
# TODO: implement

# Step 40 - adam_param_step (not yet solved)
# TODO: implement

# Step 41 - adam_step (not yet solved)
# TODO: implement

# Step 42 - init_conv_layer (not yet solved)
# TODO: implement

# Step 43 - init_linear_layer (not yet solved)
# TODO: implement

# Step 44 - init_lenet (not yet solved)
# TODO: implement

# Step 45 - forward_conv_block (not yet solved)
# TODO: implement

# Step 46 - forward_classifier_block (not yet solved)
# TODO: implement

# Step 47 - lenet_forward (not yet solved)
# TODO: implement

# Step 48 - backward_conv_block (not yet solved)
# TODO: implement

# Step 49 - backward_classifier_block (not yet solved)
# TODO: implement

# Step 50 - lenet_backward (not yet solved)
# TODO: implement

# Step 51 - lenet_predict (not yet solved)
# TODO: implement

# Step 52 - build_synthetic_image_dataset (not yet solved)
# TODO: implement

# Step 53 - shuffle_indices (not yet solved)
# TODO: implement

# Step 54 - train_test_split (not yet solved)
# TODO: implement

# Step 55 - iterate_minibatches (not yet solved)
# TODO: implement

# Step 56 - train_step (not yet solved)
# TODO: implement

# Step 57 - train_one_epoch (not yet solved)
# TODO: implement

# Step 58 - train_loop (not yet solved)
# TODO: implement

# Step 59 - evaluate (not yet solved)
# TODO: implement

