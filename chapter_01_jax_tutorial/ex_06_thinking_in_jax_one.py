from jax import jit
import jax.numpy as jnp
import numpy as np

@jit
def f(x, y):
    print("Running f():")
    print(f"  x = {x}")
    print(f"  y = {y}")
    result = jnp.dot(x + 1, y + 1)
    print(f"  result = {result}")
    return result

x = np.random.rand(3, 4)
y = np.random.rand(4)

f(x, y)

x = np.random.rand(3, 4, 4)
y = np.random.rand(4, 4)
f(x, y)


@jit
def get_negatives(x):
    filter_array = x < 0
    print(f"  filter_array = {filter_array}")
    result = x[filter_array]
    print(f"  result = {result}")
    return result

x = 1

# gets traced only once

from jax import make_jaxpr


def g(x, y):
    return jnp.dot(x + 1, y + 1)

print(make_jaxpr(g)(x, y))
