import jax
import jax.numpy as jnp
import numpy as np
from typing import Tuple, List

from jax import grad, jit

# z = xW_1 + b_1
# h = activation(h)
# y = hW_2 + b_2
from chapter_03_pieces_of_transformer.sneaky_data_generator import data_generator, batched_data_generator_jax
from lib.profile import easy_profile

batch_size = 512
no_batches = 50000
hidden_size = 128
learning_rate = 1e-3

key = jax.random.PRNGKey(1337)


def relu(x: jnp.ndarray) -> jnp.ndarray:
    return jnp.where(x > 0, x, 0)


def layer_params(
        in_dim: int,
        out_dim: int,
        key: jax.random.PRNGKey,
        scale=1e-1
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    key_W, key_b = jax.random.split(key, 2)
    W = jax.random.normal(key_W, shape=(in_dim, out_dim))
    b = jax.random.normal(key_b, shape=(out_dim,))
    return scale * W, scale * b

first_layer_key, key = jax.random.split(key, 2)
second_layer_key, key = jax.random.split(key, 2)
W_1, b_1 = layer_params(in_dim=32, out_dim=hidden_size, key=first_layer_key)
W_2, b_2 = layer_params(in_dim=hidden_size, out_dim=1, key=second_layer_key)

@jit
def predict(
        params: List[jnp.ndarray],
        x: jnp.ndarray,  # batch size x in_dim
) -> jnp.ndarray:
    W_1, b_1, W_2, b_2 = params
    z = x @ W_1 + b_1   # z is batch_size x hidden dim
    h = relu(z)
    y_hat = h @ W_2 + b_2
    return y_hat

@jit
def loss(
        params: List[jnp.ndarray],
        x: jnp.ndarray,      # batch size x in_dim
        y: jnp.ndarray,      # batch_size x out_dim
) -> jnp.ndarray:
    y_hat = predict(params, x)
    loss_per_item = (y_hat - y) ** 2
    return loss_per_item.mean()


grad_func = jit(grad(loss))
datagen_key = jax.random.PRNGKey(1337)

for batch_no in range(no_batches):
    batch_key, datagen_key = jax.random.split(datagen_key, 2)
    ts, xs, ys = batched_data_generator_jax(random_key=batch_key, batch_size=batch_size)

    params = [W_1, b_1, W_2, b_2]
    y_hat = predict(params, xs)
    loss_this_batch = loss(params, xs, ys)
    dW_1, db_1, dW_2, db_2 = grad_func(params, xs, ys)

    W_1 -= learning_rate * dW_1
    W_2 -= learning_rate * dW_2
    b_1 -= learning_rate * db_1
    b_2 -= learning_rate * db_2

    if batch_no % 500 == 0:
        print(f"Batch no = {batch_no} loss = {loss_this_batch}")



