import jax.numpy as jnp
from jax import lax

print(jnp.add(1, 1.0))
# lax requires explicit type promotion
# print(lax.add(1, 1.0)) <- crash
# print(lax.add(jnp.float16(1), 1.0)) <- crash
# jnp.float64(1) requires special extension
print(lax.add(jnp.float32(1), 1.0))

