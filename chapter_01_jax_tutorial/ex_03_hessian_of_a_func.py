
import jax.numpy as jnp
from jax import jacfwd, jacrev, jit


# R^2 -> R
def func(x, y):
    return jnp.square(x) + 3 * jnp.square(y)

def hessian(fun):
    return jacfwd(jacrev(fun))


hes = hessian(func)
x_small = jnp.arange(2.)
print(x_small)
print(hes(x_small, x_small))

# doesn't make sense!
# hessian should be [2.0, 0] [0, 6], but it's
"""
hes(jnp.zeros(2), jnp.zeros(2))
DeviceArray([[[2., 0.],
              [0., 0.]],
             [[0., 0.],
              [0., 2.]]], dtype=float32
"""


