# type: ignore
# Currently pyright doesn't support numba.cuda

from typing import Callable, Optional, TypeVar, Any

import numba
from numba import cuda
from numba.cuda import jit as _jit
from .tensor import Tensor
from .tensor_data import (
    MAX_DIMS,
    Shape,
    Storage,
    Strides,
    TensorData,
    broadcast_index,
    index_to_position,
    shape_broadcast,
    to_index,
)
from .tensor_ops import MapProto, TensorOps

FakeCUDAKernel = Any

# This code will CUDA compile fast versions your tensor_data functions.
# If you get an error, read the docs for NUMBA as to what is allowed
# in these functions.

Fn = TypeVar("Fn")


def device_jit(fn: Fn, **kwargs: Any) -> Fn:
    """Wrapper function to enable JIT compilation for device-level CUDA kernels.

    This function applies just-in-time (JIT) compilation to the provided function `fn`
    with the `device=True` argument, which signifies that the function is to be used
    as a device function in CUDA programming. Device functions are executed on a per-thread
    basis and are not callable from the host.

    Args:
    ----
        fn (Fn): The function to be JIT-compiled for device execution.
        **kwargs: Additional keyword arguments to pass to the `_jit` function.

    Returns:
    -------
        Fn: The JIT-compiled version of the input function.

    """
    return _jit(device=True, **kwargs)(fn)  # type: ignore


def jit(fn: Fn, **kwargs: Any) -> FakeCUDAKernel:
    """Wrapper function to enable JIT compilation for general CUDA kernels.

    This function applies just-in-time (JIT) compilation to the provided function `fn`
    without specifying the `device=True` argument. It is typically used for host-callable
    CUDA kernels that manage GPU threads, blocks, and grid computations.

    Args:
    ----
        fn : The function to be JIT-compiled.
        **kwargs: Additional keyword arguments to pass to the `_jit` function.

    Returns:
    -------
        FakeCUDAKernel: The JIT-compiled version of the input function as a callable CUDA kernel.

    """
    return _jit(**kwargs)(fn)  # type: ignore


to_index = device_jit(to_index)
index_to_position = device_jit(index_to_position)
broadcast_index = device_jit(broadcast_index)

THREADS_PER_BLOCK = 32


class CudaOps(TensorOps):
    cuda = True

    @staticmethod
    def map(fn: Callable[[float], float]) -> MapProto:
        """Applies a function element-wise to a tensor using CUDA.

        This method compiles the provided function `fn` to run on the GPU and applies it
        element-wise to the input tensor `a`. The result is stored in the output tensor `out`.

        Args:
        ----
            fn (Callable[[float], float]): The function to apply element-wise to the tensor.

        Returns:
        -------
            MapProto: A function that takes a tensor `a` and an optional output tensor `out`,
                      applies `fn` element-wise, and returns the result.

        """
        cufn: Callable[[float], float] = device_jit(fn)
        f = tensor_map(cufn)

        def ret(a: Tensor, out: Optional[Tensor] = None) -> Tensor:
            if out is None:
                out = a.zeros(a.shape)

            # Instantiate and run the cuda kernel.
            threadsperblock = THREADS_PER_BLOCK
            blockspergrid = (out.size + THREADS_PER_BLOCK - 1) // THREADS_PER_BLOCK
            f[blockspergrid, threadsperblock](*out.tuple(), out.size, *a.tuple())  # type: ignore
            return out

        return ret

    @staticmethod
    def zip(fn: Callable[[float, float], float]) -> Callable[[Tensor, Tensor], Tensor]:
        """Applies a function element-wise to two tensors using CUDA.

        This method compiles the provided function `fn` to run on the GPU and applies it
        element-wise to the input tensors `a` and `b`. The result is stored in the output tensor `out`.

        Args:
        ----
            fn (Callable[[float, float], float]): The function to apply element-wise to the tensors.

        Returns:
        -------
            Callable[[Tensor, Tensor], Tensor]: A function that takes two tensors `a` and `b`,
                                                applies `fn` element-wise, and returns the result.

        """
        cufn: Callable[[float, float], float] = device_jit(fn)
        f = tensor_zip(cufn)

        def ret(a: Tensor, b: Tensor) -> Tensor:
            c_shape = shape_broadcast(a.shape, b.shape)
            out = a.zeros(c_shape)
            threadsperblock = THREADS_PER_BLOCK
            blockspergrid = (out.size + (threadsperblock - 1)) // threadsperblock
            f[blockspergrid, threadsperblock](  # type: ignore
                *out.tuple(), out.size, *a.tuple(), *b.tuple()
            )
            return out

        return ret

    @staticmethod
    def reduce(
        fn: Callable[[float, float], float], start: float = 0.0
    ) -> Callable[[Tensor, int], Tensor]:
        """Reduces a tensor along a specified dimension using CUDA.

        This method compiles the provided reduction function `fn` to run on the GPU and applies it
        to reduce the input tensor `a` along the specified dimension `dim`. The result is stored
        in the output tensor `out_a`.

        Args:
        ----
            fn (Callable[[float, float], float]): The reduction function to apply.
            start (float): The initial value for the reduction.

        Returns:
        -------
            Callable[[Tensor, int], Tensor]: A function that takes a tensor `a` and a dimension `dim`,
            applies the reduction function `fn` along `dim`, and returns the result.

        """
        cufn: Callable[[float, float], float] = device_jit(fn)
        f = tensor_reduce(cufn)

        def ret(a: Tensor, dim: int) -> Tensor:
            out_shape = list(a.shape)
            out_shape[dim] = (a.shape[dim] - 1) // 1024 + 1
            out_a = a.zeros(tuple(out_shape))

            threadsperblock = 1024
            blockspergrid = out_a.size
            f[blockspergrid, threadsperblock](  # type: ignore
                *out_a.tuple(), out_a.size, *a.tuple(), dim, start
            )

            return out_a

        return ret

    @staticmethod
    def matrix_multiply(a: Tensor, b: Tensor) -> Tensor:
        """Multiplies two matrices using CUDA.

        This method performs matrix multiplication on the GPU. It ensures that the input tensors `a`
        and `b` are treated as 3-dimensional tensors for batch processing, performs the multiplication,
        and then returns the result.

        Args:
        ----
            a (Tensor): The first input tensor.
            b (Tensor): The second input tensor.

        Returns:
        -------
            Tensor: The result of the matrix multiplication.

        """
        # Make these always be a 3 dimensional multiply
        both_2d = 0
        if len(a.shape) == 2:
            a = a.contiguous().view(1, a.shape[0], a.shape[1])
            both_2d += 1
        if len(b.shape) == 2:
            b = b.contiguous().view(1, b.shape[0], b.shape[1])
            both_2d += 1
        both_2d = both_2d == 2

        ls = list(shape_broadcast(a.shape[:-2], b.shape[:-2]))
        ls.append(a.shape[-2])
        ls.append(b.shape[-1])
        assert a.shape[-1] == b.shape[-2]
        out = a.zeros(tuple(ls))

        # One block per batch, extra rows, extra col
        blockspergrid = (
            (out.shape[1] + (THREADS_PER_BLOCK - 1)) // THREADS_PER_BLOCK,
            (out.shape[2] + (THREADS_PER_BLOCK - 1)) // THREADS_PER_BLOCK,
            out.shape[0],
        )
        threadsperblock = (THREADS_PER_BLOCK, THREADS_PER_BLOCK, 1)

        tensor_matrix_multiply[blockspergrid, threadsperblock](
            *out.tuple(), out.size, *a.tuple(), *b.tuple()
        )

        # Undo 3d if we added it.
        if both_2d:
            out = out.view(out.shape[1], out.shape[2])
        return out


# Implement


def tensor_map(
    fn: Callable[[float], float],
) -> Callable[[Storage, Shape, Strides, Storage, Shape, Strides], None]:
    """CUDA higher-order tensor map function. ::

      fn_map = tensor_map(fn)
      fn_map(out, ... )

    Args:
    ----
        fn: function mappings floats-to-floats to apply.

    Returns:
    -------
        Tensor map function.

    """

    def _map(
        out: Storage,
        out_shape: Shape,
        out_strides: Strides,
        out_size: int,
        in_storage: Storage,
        in_shape: Shape,
        in_strides: Strides,
    ) -> None:
        out_index = cuda.local.array(MAX_DIMS, numba.int32)
        in_index = cuda.local.array(MAX_DIMS, numba.int32)
        i = cuda.blockIdx.x * cuda.blockDim.x + cuda.threadIdx.x
        # TODO: Implement for Task 3.3.
        if i >= out_size:
            return

        to_index(i, out_shape, out_index)
        broadcast_index(out_index, out_shape, in_shape, in_index)
        in_position = index_to_position(in_index, in_strides)
        out_position = index_to_position(out_index, out_strides)
        out[out_position] = fn(in_storage[in_position])

    return cuda.jit()(_map)  # type: ignore


def tensor_zip(
    fn: Callable[[float, float], float],
) -> Callable[
    [Storage, Shape, Strides, Storage, Shape, Strides, Storage, Shape, Strides], None
]:
    """CUDA higher-order tensor zipWith (or map2) function ::

      fn_zip = tensor_zip(fn)
      fn_zip(out, ...)

    Args:
    ----
        fn: function mappings two floats to float to apply.

    Returns:
    -------
        Tensor zip function.

    """

    def _zip(
        out: Storage,
        out_shape: Shape,
        out_strides: Strides,
        out_size: int,
        a_storage: Storage,
        a_shape: Shape,
        a_strides: Strides,
        b_storage: Storage,
        b_shape: Shape,
        b_strides: Strides,
    ) -> None:
        out_index = cuda.local.array(MAX_DIMS, numba.int32)
        a_index = cuda.local.array(MAX_DIMS, numba.int32)
        b_index = cuda.local.array(MAX_DIMS, numba.int32)
        i = cuda.blockIdx.x * cuda.blockDim.x + cuda.threadIdx.x

        # TODO: Implement for Task 3.3.
        if i >= out_size:
            return

        to_index(i, out_shape, out_index)
        broadcast_index(out_index, out_shape, a_shape, a_index)
        broadcast_index(out_index, out_shape, b_shape, b_index)
        a_position = index_to_position(a_index, a_strides)
        b_position = index_to_position(b_index, b_strides)
        out_position = index_to_position(out_index, out_strides)
        out[out_position] = fn(a_storage[a_position], b_storage[b_position])

    return cuda.jit()(_zip)  # type: ignore


def _sum_practice(out: Storage, a: Storage, size: int) -> None:
    """Practice Sum Kernel for Reduce Operation.

    This kernel demonstrates summation over blocks of elements in an input array.
    Given an input array `a` of size `n`, and an output array `out` of size `n // blockDim`,
    it computes the sum of every `blockDim` elements in `a` and stores the result in `out`.

    Example:
    -------
    Input:  [a_1, a_2, ..., a_100]  (size = 100, blockDim = 32)

    Output: [a_1 + ... + a_32, a_33 + ... + a_64, ..., a_97 + ... + a_100]

    Implementation Notes:
    ----------------------
    1. Each block of threads performs a partial sum of `blockDim` elements.
    2. Shared memory is used within each block to store intermediate results,
    ensuring efficient computation.
    3. Threads within a block collaboratively reduce their assigned elements
    to a single sum.

    Args:
    ----
        out (Storage): The storage for the output tensor, where the reduced values are stored.
        a (Storage): The storage for the input tensor to be reduced.
        size (int): The total number of elements in the input tensor `a`.

    """
    BLOCK_DIM = 32

    cache = cuda.shared.array(BLOCK_DIM, numba.float64)
    i = cuda.blockIdx.x * cuda.blockDim.x + cuda.threadIdx.x
    pos = cuda.threadIdx.x

    # TODO: Implement for Task 3.3.
    if i < size:
        cache[pos] = a[i]
    else:
        cache[pos] = 0.0
    cuda.syncthreads()

    s = BLOCK_DIM // 2
    while s >= 1:
        if pos < s:
            cache[pos] += cache[pos + s]
        cuda.syncthreads()
        s //= 2

    if pos == 0:
        out[cuda.blockIdx.x] = cache[0]


jit_sum_practice = cuda.jit()(_sum_practice)


def sum_practice(a: Tensor) -> TensorData:
    """Given a tensor `a`, this function computes the sum of all elements in the tensor."""
    (size,) = a.shape
    threadsperblock = THREADS_PER_BLOCK
    blockspergrid = (size // THREADS_PER_BLOCK) + 1
    out = TensorData([0.0 for i in range(2)], (2,))
    out.to_cuda_()
    jit_sum_practice[blockspergrid, threadsperblock](
        out.tuple()[0], a._tensor._storage, size
    )
    return out


def tensor_reduce(
    fn: Callable[[float, float], float],
) -> Callable[[Storage, Shape, Strides, Storage, Shape, Strides, int], None]:
    """CUDA higher-order tensor reduce function.

    Args:
    ----
        fn: reduction function maps two floats to float.

    Returns:
    -------
        Tensor reduce function.

    """

    def _reduce(
        out: Storage,
        out_shape: Shape,
        out_strides: Strides,
        out_size: int,
        a_storage: Storage,
        a_shape: Shape,
        a_strides: Strides,
        reduce_dim: int,
        reduce_value: float,
    ) -> None:
        BLOCK_DIM = 1024
        cache = cuda.shared.array(BLOCK_DIM, numba.float64)
        out_index = cuda.local.array(MAX_DIMS, numba.int32)
        a_index = cuda.local.array(MAX_DIMS, numba.int32)
        out_pos = cuda.blockIdx.x
        pos = cuda.threadIdx.x

        # TODO: Implement for Task 3.3.
        if out_pos >= out_size:
            return

        cache[pos] = reduce_value

        to_index(out_pos, out_shape, out_index)

        for dim in range(len(out_shape)):
            a_index[dim] = out_index[dim]

        reduce_dim_size = a_shape[reduce_dim]
        for d in range(pos, reduce_dim_size, BLOCK_DIM):
            a_index[reduce_dim] = d
            a_position = index_to_position(a_index, a_strides)
            cache[pos] = fn(numba.float64(cache[pos]), a_storage[a_position])

        cuda.syncthreads()

        s = BLOCK_DIM // 2
        while s > 0:
            if pos < s:
                cache[pos] = fn(cache[pos], cache[pos + s])
            cuda.syncthreads()
            s //= 2

        if pos == 0:
            out_pos = index_to_position(out_index, out_strides)
            out[out_pos] = cache[0]

    return jit(_reduce)  # type: ignore


def _mm_practice(out: Storage, a: Storage, b: Storage, size: int) -> None:
    """Practice Square Matrix-Matrix Multiplication (MM) Kernel

    This is a practice square MM kernel to prepare for matmul.

    Given a storage `out` and two storage `a` and `b`. Where we know
    both are shape [size, size] with strides [size, 1].

    Size is always < 32.

    Requirements:

    * All data must be first moved to shared memory.
    * Only read each cell in `a` and `b` once.
    * Only write to global memory once per kernel.

    Compute

    ```
     for i:
         for j:
              for k:
                  out[i, j] += a[i, k] * b[k, j]
    ```

    Args:
    ----
        out (Storage): storage for `out` tensor.
        a (Storage): storage for `a` tensor.
        b (Storage): storage for `b` tensor.
        size (int): size of the square

    """
    BLOCK_DIM = 32
    # TODO: Implement for Task 3.3.
    a_shared = cuda.shared.array((BLOCK_DIM, BLOCK_DIM), numba.float64)
    b_shared = cuda.shared.array((BLOCK_DIM, BLOCK_DIM), numba.float64)

    i = cuda.blockIdx.x * cuda.blockDim.x + cuda.threadIdx.x
    j = cuda.blockIdx.y * cuda.blockDim.y + cuda.threadIdx.y

    pi = cuda.threadIdx.x
    pj = cuda.threadIdx.y

    temp = numba.float64(0.0)

    for k in range((size + BLOCK_DIM - 1) // BLOCK_DIM):
        if k * BLOCK_DIM + pj < size and i < size:
            a_shared[pi, pj] = a[i * size + k * BLOCK_DIM + pj]
        else:
            a_shared[pi, pj] = 0.0

        if k * BLOCK_DIM + pi < size and j < size:
            b_shared[pi, pj] = b[(k * BLOCK_DIM + pi) * size + j]
        else:
            b_shared[pi, pj] = 0.0

        cuda.syncthreads()

        for kk in range(BLOCK_DIM):
            temp += a_shared[pi, kk] * b_shared[kk, pj]

        cuda.syncthreads()

    if i < size and j < size:
        out[i * size + j] = temp


jit_mm_practice = jit(_mm_practice)


def mm_practice(a: Tensor, b: Tensor) -> TensorData:
    """Given two square matrices `a` and `b`, this function computes their matrix product."""
    (size, _) = a.shape
    threadsperblock = (THREADS_PER_BLOCK, THREADS_PER_BLOCK)
    blockspergrid = 1
    out = TensorData([0.0 for i in range(size * size)], (size, size))
    out.to_cuda_()
    jit_mm_practice[blockspergrid, threadsperblock](
        out.tuple()[0], a._tensor._storage, b._tensor._storage, size
    )
    return out


def _tensor_matrix_multiply(
    out: Storage,
    out_shape: Shape,
    out_strides: Strides,
    out_size: int,
    a_storage: Storage,
    a_shape: Shape,
    a_strides: Strides,
    b_storage: Storage,
    b_shape: Shape,
    b_strides: Strides,
) -> None:
    """CUDA tensor matrix multiply function.

    Requirements:

    * All data must be first moved to shared memory.
    * Only read each cell in `a` and `b` once.
    * Only write to global memory once per kernel.

    Should work for any tensor shapes that broadcast as long as ::

    ```python
    assert a_shape[-1] == b_shape[-2]
    ```
    Returns:
        None : Fills in `out`
    """
    # Ensure inner dimensions match for matrix multiplication
    if a_shape[-1] != b_shape[-2]:
        return

    a_batch_stride = a_strides[0] if a_shape[0] > 1 else 0
    b_batch_stride = b_strides[0] if b_shape[0] > 1 else 0
    # Batch dimension - fixed
    batch = cuda.blockIdx.z

    BLOCK_DIM = 32
    a_shared = cuda.shared.array((BLOCK_DIM, BLOCK_DIM), numba.float64)
    b_shared = cuda.shared.array((BLOCK_DIM, BLOCK_DIM), numba.float64)

    # The final position c[i, j]
    i = cuda.blockIdx.x * cuda.blockDim.x + cuda.threadIdx.x
    j = cuda.blockIdx.y * cuda.blockDim.y + cuda.threadIdx.y

    # The local position in the block.
    pi = cuda.threadIdx.x
    pj = cuda.threadIdx.y

    # Code Plan:
    # 1) Move across shared dimension by block dim.
    #    a) Copy into shared memory for a matrix.
    #    b) Copy into shared memory for b matrix
    #    c) Compute the dot produce for position c[i, j]
    # TODO: Implement for Task 3.4.

    # Boundary check for thread
    if i >= out_shape[-2] or j >= out_shape[-1]:
        return

    # Code for matrix multiplication
    acc = 0.0

    # Loop over the sub-matrices
    for k_block in range((a_shape[-1] + BLOCK_DIM - 1) // BLOCK_DIM):
        # Load data into shared memory
        k = k_block * BLOCK_DIM + pj

        # Load matrix a
        if i < a_shape[-2] and k < a_shape[-1]:
            a_idx = (
                batch * a_batch_stride  # batch dimension
                + i * a_strides[-2]  # row dimension
                + k * a_strides[-1]  # column dimension
            )
            a_shared[pi, pj] = a_storage[a_idx]
        else:
            a_shared[pi, pj] = 0.0

        # Load matrix b
        k = k_block * BLOCK_DIM + pi
        if k < b_shape[-2] and j < b_shape[-1]:
            b_idx = (
                batch * b_batch_stride  # batch dimension
                + k * b_strides[-2]  # row dimension
                + j * b_strides[-1]  # column dimension
            )
            b_shared[pi, pj] = b_storage[b_idx]
        else:
            b_shared[pi, pj] = 0.0

        # Synchronize threads
        cuda.syncthreads()

        # Compute partial dot product
        for k in range(BLOCK_DIM):
            if k_block * BLOCK_DIM + k < a_shape[-1]:  # Check tile boundary
                acc += a_shared[pi, k] * b_shared[k, pj]

        # Synchronize before loading next sub-matrix
        cuda.syncthreads()

    # Write the result
    out_idx = (
        batch * out_strides[0]  # batch dimension
        + i * out_strides[-2]  # row dimension
        + j * out_strides[-1]  # column dimension
    )
    out[out_idx] = acc


tensor_matrix_multiply = jit(_tensor_matrix_multiply)
