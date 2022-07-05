import numpy as np
import matplotlib.pyplot as plt

"""
Task: Predict (learn) function y = sin(x)
Take pairs x, y = sin(x)
y_hat = nnet(x)
"""

batch_size = 256
init_size = 1e-1


hidden_size = 128

def relu(x):
    return np.where(x > 0, x, 0)

def d_relu(x):
    return np.where(x > 0, 1, 0)

# z = A_1 x + b_1
# h = relu(z)
# y_hat = A_2 * h + b_2

A_1 = np.random.randn(1, hidden_size) * init_size
b_1 = np.random.randn(hidden_size)

A_2 = np.random.randn(hidden_size, 1) * init_size
b_2 = np.random.randn(1)

no_batches = 100000
learning_rate = 1e-3
report_every = 1000

for t in range(no_batches + 1):
    # draw a batch
    x = np.random.uniform(-4., 4, size=(batch_size, 1))
    y = np.sin(x)

    # forward pass
    z = x @ A_1 + b_1
    h = relu(z)
    y_hat = h @ A_2 + b_2

    # compute loss

    loss = 0.5 * (y_hat - y) ** 2

    # backward pass
    d_loss_d_y_hat = y_hat - y
    d_loss_d_b_2 = d_loss_d_y_hat.mean(0)
    d_loss_d_A_2 = np.einsum('bh,bo->bho', h, d_loss_d_y_hat).mean(0)

    d_loss_d_h = np.einsum('bo,ho->bh', d_loss_d_y_hat, A_2)
    d_loss_d_z = d_relu(z) * d_loss_d_h

    d_loss_d_A_1 = np.einsum('bh,bo->boh', d_loss_d_z, x).mean(0)
    d_loss_d_b_1 = d_loss_d_z.mean(0)

    # optimization step
    A_1 -= learning_rate * d_loss_d_A_1
    b_1 -= learning_rate * d_loss_d_b_1
    A_2 -= learning_rate * d_loss_d_A_2
    b_2 -= learning_rate * d_loss_d_b_2

    if t % report_every == 0:
        print(f"{loss.mean()=:.3f}")
        xs = np.linspace(-4., 4., num=100).reshape(-1, 1)
        ys = np.sin(xs)
        ys_hat = relu(xs @ A_1 + b_1) @ A_2 + b_2
        plt.plot(xs, ys, color='red')
        plt.plot(xs, ys_hat, color='green')
        plt.title(f"Prediction at {t=}")
        plt.show()






