"""Collection of the core mathematical operators used throughout the code base."""


# ## Task 0.1

#
# Implementation of a prelude of elementary functions.

# Mathematical functions:
# - mul
# - id
# - add
# - neg
# - lt
# - eq
# - max
# - is_close
# - sigmoid
# - relu
# - log
# - exp
# - log_back
# - inv
# - inv_back
# - relu_back
#
# For sigmoid calculate as:
# $f(x) =  \frac{1.0}{(1.0 + e^{-x})}$ if x >=0 else $\frac{e^x}{(1.0 + e^{x})}$
# For is_close:
# $f(x) = |x - y| < 1e-2$


# TODO: Implement for Task 0.1.

import math
from typing import Iterable, Callable


def mul(a: float, b: float) -> float:
    """Multiply two numbers.

    Args:
    ----
        a (float): The first number.
        b (float): The second number.

    Returns:
    -------
        float: The product of the two numbers.

    """
    return a * b


def id(a: float) -> float:
    """Identity function.

    Args:
    ----
        a (float): The input value.

    Returns:
    -------
        float: The input value.

    """
    return a


def add(a: float, b: float) -> float:
    """Add two numbers.

    Args:
    ----
        a (float): The first number.
        b (float): The second number.

    Returns:
    -------
        float: The sum of the two numbers.

    """
    return a + b


def neg(a: float) -> float:
    """Negate a number.

    Args:
    ----
        a (float): The input value.

    Returns:
    -------
        float: The negation of the input value.

    """
    return -a


def lt(a: float, b: float) -> bool:
    """Less than comparison.

    Args:
    ----
        a (float): The first number.
        b (float): The second number.

    Returns:
    -------
        bool: True if a is less than b, False otherwise.

    """
    return a < b


def eq(a: float, b: float) -> bool:
    """Equality comparison.

    Args:
    ----
        a (float): The first number.
        b (float): The second number.

    Returns:
    -------
        bool: True if a is equal to b, False otherwise.

    """
    return a == b


def max(a: float, b: float) -> float:
    """Maximum of two numbers.

    Args:
    ----
        a (float): The first number.
        b (float): The second number.

    Returns:
    -------
        float: The maximum of the two numbers.

    """
    return a if a > b else b


def is_close(a: float, b: float) -> bool:
    """Check if two numbers are close.

    Args:
    ----
        a (float): The first number.
        b (float): The second number.

    Returns:
    -------
        bool: True if the numbers are within 1e-2, False otherwise.

    """
    return abs(a - b) < 1e-2


def sigmoid(a: float) -> float:
    """Sigmoid function.

    Args:
    ----
        a (float): The input value.

    Returns:
    -------
        float: The sigmoid of the input value.

    """
    if a >= 0:
        return 1.0 / (1.0 + math.exp(-a))
    else:
        return math.exp(a) / (1.0 + math.exp(a))


def relu(a: float) -> float:
    """ReLU function.

    Args:
    ----
        a (float): The input value.

    Returns:
    -------
        float: The ReLU of the input value.

    """
    return a if a > 0 else 0


def log(a: float) -> float:
    """Log function.

    Args:
    ----
        a (float): The input value.

    Returns:
    -------
        float: The log of the input value.

    """
    return math.log(a)


def exp(a: float) -> float:
    """Exponential function.

    Args:
    ----
        a (float): The input value.

    Returns:
    -------
        float: The exponential of the input value.

    """
    return math.exp(a)


def inv(a: float) -> float:
    """Inverse function.

    Args:
    ----
        a (float): The input value.

    Returns:
    -------
        float: The inverse of the input value.

    """
    return 1.0 / a


def log_back(a: float, b: float) -> float:
    """Computes the derivative of log(a) multiplied by a second argument b.

    Args:
    ----
        a (float): The input value to the log function.
        b (float): The upstream gradient or another multiplier.

    Returns:
    -------
        float: The result of (1/a) * b, which is the derivative of log(a) times b.

    """
    return (1 / a) * b


def inv_back(a: float, b: float) -> float:
    """Computes the derivative of 1/a multiplied by a second argument b.

    Args:
    ----
        a (float): The input value to the inverse function.
        b (float): The upstream gradient or another multiplier.

    Returns:
    -------
        float: The result of (-1/(a^2)) * b, which is the derivative of 1/a times b.

    """
    return (-1 / (a**2)) * b


def relu_back(a: float, b: float) -> float:
    """Computes the derivative of the ReLU function multiplied by a second argument b.

    Args:
    ----
        a (float): The input value to the ReLU function.
        b (float): The upstream gradient or another multiplier.

    Returns:
    -------
        float: The result of (1 if a > 0 else 0) times b.

    """
    return (1 if a > 0 else 0) * b


# ## Task 0.3

# Small practice library of elementary higher-order functions.

# Implement the following core functions
# - map
# - zipWith
# - reduce
#
# Use these to implement
# - negList : negate a list
# - addLists : add two lists together
# - sum: sum lists
# - prod: take the product of lists

# TODO: Implement for Task 0.3.


def map(fn: Callable[[float], float], items: Iterable[float]) -> list[float]:
    """Apply a function to each item in an iterable.

    Args:
    ----
        fn (function): The function to apply.
        items (iterable): The iterable of items.

    Returns:
    -------
        list: The list of items after applying the function to each item.

    """
    return [fn(item) for item in items]


def customzip(
    items1: Iterable[float], items2: Iterable[float]
) -> list[tuple[float, float]]:
    """Zip two iterables together.

    Args:
    ----
        items1 (Iterable[float]): The first iterable of items.
        items2 (Iterable[float]): The second iterable of items.

    Returns:
    -------
        list[tuple[float, float]]: The list of tuples after zipping the two iterables together.

    """
    # Create iterators for both iterables
    iter1 = iter(items1)
    iter2 = iter(items2)

    result = []

    while True:
        try:
            a = next(iter1)
            b = next(iter2)
            result.append((a, b))
        except StopIteration:
            break

    return result


def zipWith(
    fn: Callable[[float, float], float],
    items1: Iterable[float],
    items2: Iterable[float],
) -> list[float]:
    """Apply a function to pairs of items from two iterables.

    Args:
    ----
        fn (Callable[[float, float], float]): The function to apply.
        items1 (Iterable[float]): The first iterable of items.
        items2 (Iterable[float]): The second iterable of items.

    Returns:
    -------
        list[float]: The list of items after applying the function to each pair of items.

    """
    return [fn(a, b) for a, b in customzip(items1, items2)]


def reduce(fn: Callable[[float, float], float], items: Iterable[float]) -> float:
    """Reduce an iterable of items to a single value using a binary function.

    Args:
    ----
        fn (Callable[[float, float], float]): The function to apply.
        items (Iterable[float]): The iterable of items to reduce.

    Returns:
    -------
        float: The result of reducing the iterable of items to a single value.
        Returns 0 if empty.

    """
    iterator = iter(items)
    try:
        result = next(iterator)
    except StopIteration:
        return 0.0
    for item in iterator:
        result = fn(result, item)
    return result


def negList(items: Iterable[float]) -> Iterable[float]:
    """Negate each item in an iterable.

    Args:
    ----
        items (iterable): The iterable of items.

    Returns:
    -------
        list[float]: The list of items after negating each item.

    """
    return list(map(neg, items))


def addLists(items1: Iterable[float], items2: Iterable[float]) -> Iterable[float]:
    """Add corresponding items in two iterables together.

    Args:
    ----
        items1 (iterable): The first iterable of items.
        items2 (iterable): The second iterable of items.

    Returns:
    -------
        list[float]: The list of items after adding corresponding items in the two iterables together.

    """
    return zipWith(add, items1, items2)


def sum(items: Iterable[float]) -> float:
    """Sum an iterable of items.

    Args:
    ----
        items (iterable): The iterable of items.

    Returns:
    -------
        float: The sum of the items in the iterable.

    """
    return reduce(add, items)


def prod(items: Iterable[float]) -> float:
    """Take the product of an iterable of items.

    Args:
    ----
        items (iterable): The iterable of items.

    Returns:
    -------
        float: The product of the items in the iterable.

    """
    return reduce(mul, items)
