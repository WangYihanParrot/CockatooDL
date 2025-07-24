from cockatoo.tensor import Tensor

def mse_loss(y_pred: Tensor, y_true: Tensor) -> Tensor:
    diff = y_pred - y_true
    sqr = diff ** 2
    return sqr.mean()
