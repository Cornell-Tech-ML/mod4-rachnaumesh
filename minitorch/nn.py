from typing import Tuple

from .autodiff import Context
from .tensor import Tensor
from .tensor_functions import Function, rand


# List of functions in this file:
# - avgpool2d: Tiled average pooling 2D
# - argmax: Compute the argmax as a 1-hot tensor
# - Max: New Function for max operator
# - max: Apply max reduction
# - softmax: Compute the softmax as a tensor
# - logsoftmax: Compute the log of the softmax as a tensor - See https://en.wikipedia.org/wiki/LogSumExp#log-sum-exp_trick_for_log-domain_calculations
# - maxpool2d: Tiled max pooling 2D
# - dropout: Dropout positions based on random noise, include an argument to turn off


def tile(input: Tensor, kernel: Tuple[int, int]) -> Tuple[Tensor, int, int]:
    """Reshape an image tensor for 2D pooling

    Args:
    ----
        input: batch x channel x height x width
        kernel: height x width of pooling

    Returns:
    -------
        Tensor of size batch x channel x new_height x new_width x (kernel_height * kernel_width) as well as the new_height and new_width value.

    """
    batch, channel, height, width = input.shape
    kh, kw = kernel
    assert height % kh == 0
    assert width % kw == 0
    # TODO: Implement for Task 4.3.
    new_height = height // kh
    new_width = width // kw
    tile_size = kh * kw
    output = input.contiguous().view(batch, channel, new_height, kh, new_width, kw)
    output = output.permute(0, 1, 2, 4, 3, 5)
    output = output.contiguous().view(batch, channel, new_height, new_width, tile_size)
    return output, new_height, new_width


# TODO: Implement for Task 4.3.
def avgpool2d(input: Tensor, kernel: Tuple[int, int]) -> Tensor:
    """Tiled average pooling 2D.

    Args:
    ----
        input: Tensor of size batch x channel x height x width
        kernel: height x width of pooling

    Returns:
    -------
        Tensor of size batch x channel x new_height x new_width

    """
    output, new_height, new_width = tile(input, kernel)
    # Take mean over the last dimension (tile_size)
    pooled = output.mean(4)
    return pooled.contiguous().view(
        output.shape[0], output.shape[1], new_height, new_width
    )


def argmax(input: Tensor, dim: int) -> Tensor:
    """Compute the argmax as a 1-hot tensor

    Args:
    ----
        input: Tensor
        dim: int

    Returns:
    -------
        Tensor: 1-hot tensor

    """
    # Create a mask where the maximum value is 1 and others are 0
    max_vals = input.f.max_reduce(input, dim)
    return input.f.eq_zip(input, max_vals)


class Max(Function):
    @staticmethod
    def forward(ctx: Context, input: Tensor, dim: Tensor) -> Tensor:
        """Compute the max along a dimension"""
        output = input.f.max_reduce(input, int(dim.item()))
        mask = input.f.eq_zip(input, output)
        ctx.save_for_backward(mask)
        return output

    @staticmethod
    def backward(ctx: Context, grad_output: Tensor) -> Tuple[Tensor, float]:
        """Compute the gradient of the max"""
        (mask,) = ctx.saved_values
        return mask * grad_output, 0.0


def max(input: Tensor, dim: int) -> Tensor:
    """Compute the max along a dimension.

    Args:
    ----
        input: Tensor
        dim: dimension to compute max

    Returns:
    -------
        Tensor with dimension dim reduced to 1

    """
    if dim is None:
        return Max.apply(input.contiguous().view(input.size), input._ensure_tensor(0))
    else:
        return Max.apply(input, input._ensure_tensor(dim))


def softmax(input: Tensor, dim: int) -> Tensor:
    """Compute the softmax as a tensor

    Args:
    ----
        input: Tensor
        dim: int

    Returns:
    -------
        Tensor: softmax tensor

    """
    # Subtract max for numerical stability
    out = input.exp()
    return out / (out.sum(dim))


def logsoftmax(input: Tensor, dim: int) -> Tensor:
    """Compute the log of the softmax as a tensor

    Args:
    ----
        input: Tensor
        dim: int

    Returns:
    -------
        Tensor: log of the softmax tensor

    """
    max_vals = max(input, dim)
    shifted = input - max_vals
    exp_vals = shifted.exp()
    sum_exp = exp_vals.sum(dim)
    log_sum_exp = sum_exp.log() + max_vals
    return input - log_sum_exp


def maxpool2d(input: Tensor, kernel: Tuple[int, int]) -> Tensor:
    """Tiled max pooling 2D.

    Args:
    ----
        input: Tensor of size batch x channel x height x width
        kernel: height x width of pooling

    Returns:
    -------
        Tensor of size batch x channel x new_height x new_width

    """
    output, new_height, new_width = tile(input, kernel)
    # Take max over the last dimension (tile_size)
    pooled = max(output, 4)
    return pooled.contiguous().view(
        output.shape[0], output.shape[1], new_height, new_width
    )


def dropout(input: Tensor, p: float, ignore: bool = False) -> Tensor:
    """Dropout positions based on random noise, include an argument to turn off

    Args:
    ----
        input: Tensor
        p: float
        ignore: bool

    Returns:
    -------
        Tensor: with dropout applied

    """
    if ignore or p == 0.0:
        return input

    if p == 1.0:
        return input * 0.0

    # Create dropout mask
    mask = rand(input.shape) > p
    # Scale the output by 1/(1-p) to maintain expected value
    scale = 1.0 / (1.0 - p)
    return input * mask * scale
