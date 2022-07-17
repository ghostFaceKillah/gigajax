from typing import Tuple
from lib.types import NumpyArray

import numpy as np
import jax.numpy as jnp
import jax

import matplotlib.pyplot as plt

N = 32


def data_generator(
        rng: np.random.RandomState,
        at_x: float = 3.14,
        n: int = N,
        start: float = -3.,
        stop: float = 3.,
) -> Tuple[
    NumpyArray['n', np.float32],
    NumpyArray['n', np.float32],
    float
]:
    pre_x = np.linspace(start=start, stop=stop, num=n)

    a = np.random.uniform(-5., 5.)
    b = np.random.uniform(-5., 5.)
    c = np.random.uniform(-.5, .5)

    f = lambda x: np.sin(a * x + b) + c

    t = pre_x
    x = f(pre_x)
    y = f(at_x)

    return t, x, y


def batched_data_generator_jax(
        random_key: jax.random.PRNGKey,
        batch_size: int,
        at_x: float = 3.14,
        n: int = N,
        start: float = -3.,
        stop: float = 3.,

):
    pre_xs = jnp.tile(jnp.linspace(start=start, stop=stop, num=n), (batch_size, 1))
    at_xs = jnp.tile(at_x, (batch_size, 1))

    ak, bk, ck = jax.random.split(random_key, 3)
    a = jax.random.uniform(ak, minval=-5., maxval=5., shape=(batch_size, 1))
    b = jax.random.uniform(bk, minval=-5., maxval=5., shape=(batch_size, 1))
    c = jax.random.uniform(ck, minval=-.5, maxval=.5, shape=(batch_size, 1))

    f = lambda x: jnp.sin(a * x + b) + c

    ts = pre_xs
    xs = f(pre_xs)
    ys = at_xs

    return ts, xs, ys


def _visualize_some_data_from_the_generator():
    for color in 'bgrcmyk':
        t, x, y = data_generator(np.random.RandomState())
        plt.plot(t, x, color)
        plt.xlim(-5., 5.)
        plt.ylim(-2., 2.)
    plt.show()


if __name__ == '__main__':
    # _visualize_some_data_from_the_generator()
    batched_data_generator_jax(
        random_key=jax.random.PRNGKey(1337),
        batch_size=32
    )

