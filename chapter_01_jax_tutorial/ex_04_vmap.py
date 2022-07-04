from chapter_01_jax_tutorial.ex_01_jit_etc import timeit
import jax.numpy as jnp
from jax import random, jit, vmap

key = random.PRNGKey(0)
mat = random.normal(key, (150, 100))
batched_x = random.normal(key, (50, 100))


def apply_matrix(v):
    return jnp.dot(mat, v)


def naively_batched_apply_matrix(v_batched):
    return jnp.stack([apply_matrix(v) for v in v_batched])


@jit
def batched_apply_matrix(v_batched):
    return jnp.dot(v_batched, mat.T)


@jit
def vmap_batched_apply_matrix(v_batched):
    return vmap(apply_matrix)(v_batched)

print("Naively batched")
timeit(lambda: naively_batched_apply_matrix(batched_x).block_until_ready())


print("Manually batched")
timeit(lambda: batched_apply_matrix(batched_x).block_until_ready())


print("Auto-vectorized with vmap")
timeit(lambda: vmap_batched_apply_matrix(batched_x).block_until_ready())

