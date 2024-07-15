import ipyparallel as ipp
import numpy as np
import torch
import networkx as nx  # nx will be used for network topology creating and plotting


rc = ipp.Client(profile="bluefog")
num_nodes = len(rc.ids)
dview = rc[:]  # A DirectView of all engines
dview.block = True
rc.ids

import numpy as np
import bluefog.torch as bf
import torch
from bluefog.common import topology_util
import networkx as nx
from sklearn.datasets import make_classification
from sklearn.preprocessing import normalize

bf.init()
print(f"Hello, I am {bf.rank()} among {bf.size()} processes")

from torchvision import datasets

dataset = datasets.MNIST("../data", train=True, download=True)

idx = (dataset.targets == 0) | (dataset.targets == 1)
X = dataset.data[idx].view(dataset.data[idx].shape[0], -1)
y = dataset.targets[idx][:, None]

print("Feature X has the shape:", X.shape)
print("Label y has the shape:", y.shape)
print("Lable y should only contains labels 0 and 1:", y[:10].t())

X = X / 255.0

M, dim = X.shape
num_split = int(M // num_nodes)
for i in range(num_nodes):
    X_worker = X[i * num_split : (i + 1) * num_split]
    y_worker = y[i * num_split : (i + 1) * num_split]
    y_worker = torch.squeeze(y_worker)
    y_worker = y_worker.type(torch.FloatTensor)
    #     X_worker = X_worker.to(torch.double)
    #     y_worker = y_worker.to(torch.double)
    dview.push({"X_worker": X_worker, "y_worker": y_worker}, targets=i)


print(X_worker.shape)
print(y_worker.shape)

def sigmoid(x):
    """Return the output of the sigmoid function.
    Args:
     - x (torch tensor): Input that we want to apply the sigmoid function on.
    Return:
    - Return a torch tensor of the output

    """
    return 1.0 / (1.0 + torch.exp(-x))


def obj(w, X, y, reg):
    """Output of the objective function of logistic regression.
    Args:
     - w (torch tensor): Parameter.
     - X (torch tensor): Data.
     - y (torch tensor): Labels.
     - reg (float): Regularization parameter.
    Return:
     - Returns a float of the value of the objective function.
    """
    M, _ = X.shape

    return -1 / M * torch.sum(
        y[:, None] * X @ w - torch.log(1 + torch.exp(X @ w))
    ) + reg / 2 * torch.norm(w, p=2)


def grad(w, X, y, reg):
    """Output of the gradient for the logistic regression.
    Args:
     - w (torch tensor): Parameter.
     - X_batch (torch tensor): Batch of data gradient is calculated on.
     - y_batch (torch tensor): Labels.
     - reg (float): Regularization parameter.
    Return:
     - Returns a torch tensor that is the gradient for the data and parameters provided.
    """
    M, _ = X.shape
    return -1 / M * (X.T @ y[:, None] - (X.T @ (1 - sigmoid(-X @ w)))) + reg * w

from ipyparallel.datapub import publish_data

# Create a distributed gradient descent function that is equivalent to centralized gradient descent.
def distributed_grad_descent(X, y, reg=1e-4, maxite=1000, step_size=1):
    M, dim = X.shape
    w_opt = 0.001 * torch.randn((dim, 1))
    for i in range(maxite):
        # Calculate local gradient.
        grad_iter = grad(w_opt, X, y, reg)
        # Use local gradient to calculate global gradient.
        gradient = bf.allreduce(grad_iter)

        # Distributed gradient descent.
        w_opt = w_opt - step_size * gradient
        if i % 20 == 0:
            publish_data(dict(i=i, gradient_norm=torch.norm(gradient, p=2)))
    publish_data(
        dict(i=maxite - 1, gradient_norm=torch.norm(gradient, p=2), w_opt=w_opt)
    )
    print(f"[DG] Rank {bf.rank()}: Global Gradient Norm: {torch.norm(gradient, p=2)}")
    return w_opt

def ATC(X, y, w_opt, w=None, step_size=1.0, reg=1e-4):
    """
    Adapt then combine algorithm using blocking communication. Performs one iteration.
    Inputs:
    - X (torch tensor) := The data to calculate the loss on of size (N, dim).
    - y (torch tensor) := The corresponding labels of the data of size (N, 1).
    - w_opt (torch tensor) := The optimal parameters.
    - w (torch tensor) := The parameter to update over.
    - step_size (float) := The step size parameter.
    - reg (float) := The regularization parameter.
    - batch_size (int) := Size of a batch for the iteration.
    Output:
    - w (torch tensor) := Update iteration.
    - Relative error.
    """
    M, dim = X.shape

    if w is None:
        w = 0.001 * torch.randn((dim, 1))

    adapt = w - step_size * grad(w, X, y, reg)
    combine = bf.neighbor_allreduce(adapt)
    w = combine

    dist = torch.norm(w - w_opt, p=2) / torch.norm(w_opt, p=2)
    # averaged_relative_error = bf.allreduce(dist**2, average=True)

    return w, dist  # torch.sqrt(averaged_relative_error)
def AWC(X, y, w_opt, w=None, step_size=1.0, reg=1e-4):
    """
    Adapt with combine algorithm using non-blocking communication. Performs one iteration.
    Inputs:
    - X (torch tensor) := The data to calculate the loss on of size (N, dim).
    - y (torch tensor) := The corresponding labels of the data of size (N, 1).
    - w_opt (torch tensor) := The optimal parameters.
    - w (torch tensor) := The parameter to update over.
    - step_size (float) := The step size parameter.
    - reg (float) := The regularization parameter.
    - batch_size (int) := Size of a batch for the iteration.
    Output:
    - w (torch tensor) := Update iteration.
    - Relative error.
    """
    M, dim = X.shape

    if w is None:
        w = 0.001 * torch.randn((dim, 1))

    combine_handle = bf.neighbor_allreduce_nonblocking(w)
    adapt = step_size * grad(w, X, y, reg)
    combine = bf.wait(combine_handle)
    w = combine - adapt

    dist = torch.norm(w - w_opt, p=2) / torch.norm(w_opt, p=2)
    # averaged_relative_error = bf.allreduce(dist**2, average=True)

    return w, dist  # torch.sqrt(averaged_relative_error)

from tqdm import tqdm

maxite = 5000
dview["maxite"] = maxite
ar = dview.apply_async(
    lambda: distributed_grad_descent(X_worker, y_worker, maxite=maxite)
)
i = 0

# This may take a while finish it.
with tqdm(total=maxite - 1, desc="Epoch     #{}".format(i + 1)) as t:
    while not ar.ready():
        data = ar.data[0]
        t.set_postfix({"gradient_norm": data.get("gradient_norm", 1)})
        t.update(data.get("i", 0) - t.n)
w_opt = ar.data[0]["w_opt"]
dview["w_opt"] = w_opt

from ipyparallel.datapub import publish_data

# Create a distributed gradient descent function that is equivalent to centralized gradient descent.
def decentralized_grad_descent(
    one_step_func, X, y, w_opt, maxite, step_size=1.0, reg=1e-4
):
    print(maxite)
    M, dim = X.shape
    w = 0.001 * torch.randn((dim, 1))
    for i in range(maxite):
        w, relative_error = one_step_func(
            X, y, w=w, w_opt=w_opt, step_size=step_size, reg=reg
        )
        if i % 20 == 0:
            publish_data(dict(i=i, relative_error=relative_error, w=w))
    publish_data(dict(i=maxite - 1, relative_error=relative_error, w=w))


     
