import jax.numpy as jnp
from jax import device_put
import numpy as np
from time import perf_counter
from contextlib import contextmanager
from jax import grad, jit, vmap
from jax import random

@contextmanager
def catchtime() -> float:
    start = perf_counter()
    yield lambda: perf_counter() - start

def timeit(f):
    times = []
    for i in range(10):
        with catchtime() as t:
            f()
        times.append(t())

    print(f"execution stats = {jnp.mean(jnp.array(times))} "
          f"+/- {jnp.std(jnp.array(times))}")


if __name__ == '__main__':
    key = random.PRNGKey(0)

    x = random.normal(key, (10, ))

    print(x)


    size = 3000
    x = random.normal(key, (size, size), dtype=jnp.float32)




    x = np.random.normal(size=(size, size)).astype(np.float32)
    x = device_put(x)

    # timeit(lambda: jnp.dot(x, x.T).block_until_ready())


    def selu(x, alpha=1.67, lmbda=1.05):
        return lmbda * jnp.where(x > 0, x, alpha * jnp.exp(x) - alpha)

    def selu_np(x, alpha=1.67, lmbda=1.05):
        return lmbda * np.where(x > 0, x, alpha * np.exp(x) - alpha)

    x = random.normal(key, (100000000,))
    nx = np.random.normal(size=100000000).astype(np.float32)

    print("jax wrapped numpy")
    timeit(lambda: selu(x).block_until_ready())
    selu_jit = jit(selu)

    print("jax jit XLA")
    timeit(lambda: selu_jit(x).block_until_ready())

    print("numpy ")
    timeit(lambda: selu_np(nx))


