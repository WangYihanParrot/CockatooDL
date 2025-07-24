import numpy as np


class Tensor:
    def __init__(
            self,
            data: np.ndarray | int | float | tuple | list,
            requires_grad: bool = True,
    ):
        self.data = np.array(data, dtype=np.float32)
        self.requires_grad = requires_grad

        self.grad = None
        self._backward = lambda: None  # 当前节点的反向传播函数
        self._prev = set()  # 记录依赖的子节点，构建计算图

    # 属性
    @property
    def shape(self):
        return self.data.shape

    @property
    def dtype(self):
        return self.data.dtype

    def __repr__(self):
        return (
            f"Tensor(shape={self.shape}, data={self.data}, dtype={self.dtype}, "
            f"requires_grad={self.requires_grad}, grad={self.grad})"
        )

    # ====== 基本操作（带自动求导） ======

    def _unbroadcast(self, grad, shape):
        while len(shape) < len(grad.shape):
            shape = (1,) + shape
        axes = [i for i, (g_dim, s_dim) in enumerate(zip(grad.shape, shape)) if s_dim == 1]
        grad = grad.sum(axis=tuple(axes), keepdims=True)
        grad = grad.reshape(shape)
        return grad

    def __add__(self, other):
        other = other if isinstance(other, Tensor) else Tensor(other)
        out = Tensor(self.data + other.data, requires_grad=self.requires_grad or other.requires_grad)

        def _backward():
            if self.requires_grad:
                grad = out.grad
                self.grad = self.grad + grad if self.grad is not None else grad
            if other.requires_grad:
                grad = out.grad
                if other.data.shape != out.grad.shape:
                    grad = self._unbroadcast(grad, other.data.shape)
                other.grad = other.grad + grad if other.grad is not None else grad

        out._backward = _backward
        out._prev = {self, other}
        return out


    def __mul__(self, other):
        other = other if isinstance(other, Tensor) else Tensor(other)
        out = Tensor(self.data * other.data, requires_grad=self.requires_grad or other.requires_grad)

        def _backward():
            if self.requires_grad:
                grad = out.grad * other.data
                if grad.shape != self.data.shape:
                    grad = self._unbroadcast(grad, self.data.shape)
                self.grad = self.grad + grad if self.grad is not None else grad
            if other.requires_grad:
                grad = out.grad * self.data
                if grad.shape != other.data.shape:
                    grad = self._unbroadcast(grad, other.data.shape)
                other.grad = other.grad + grad if other.grad is not None else grad

        out._backward = _backward
        out._prev = {self, other}
        return out

    def __matmul__(self, other):
        other = other if isinstance(other, Tensor) else Tensor(other)
        out = Tensor(self.data @ other.data, requires_grad=self.requires_grad or other.requires_grad)

        def _backward():
            if self.requires_grad:
                grad = out.grad @ other.data.T
                self.grad = self.grad + grad if self.grad is not None else grad
            if other.requires_grad:
                grad = self.data.T @ out.grad
                other.grad = other.grad + grad if other.grad is not None else grad

        out._backward = _backward
        out._prev = {self, other}
        return out

    def __neg__(self):
        out = Tensor(-self.data, requires_grad=self.requires_grad)

        def _backward():
            if self.requires_grad:
                grad = -out.grad
                self.grad = self.grad + grad if self.grad is not None else grad

        out._backward = _backward
        out._prev = {self}
        return out

    def __sub__(self, other):
        other = other if isinstance(other, Tensor) else Tensor(other)
        return self + (-other)

    def __truediv__(self, other):
        other = other if isinstance(other, Tensor) else Tensor(other)
        out = Tensor(self.data / other.data, requires_grad=self.requires_grad or other.requires_grad)

        def _backward():
            if self.requires_grad:
                grad = out.grad / other.data
                self.grad = self.grad + grad if self.grad is not None else grad
            if other.requires_grad:
                grad = -out.grad * self.data / (other.data ** 2)
                other.grad = other.grad + grad if other.grad is not None else grad

        out._backward = _backward
        out._prev = {self, other}
        return out

    def __pow__(self, power):
        assert isinstance(power, (int, float)), "只支持常数幂"
        out = Tensor(self.data ** power, requires_grad=self.requires_grad)

        def _backward():
            if self.requires_grad:
                grad = out.grad * power * (self.data ** (power - 1))
                self.grad = self.grad + grad if self.grad is not None else grad

        out._backward = _backward
        out._prev = {self}
        return out

    def mean(self):
        out = Tensor(np.mean(self.data), requires_grad=self.requires_grad)

        def _backward():
            if self.requires_grad:
                grad = out.grad * np.ones_like(self.data) / self.data.size
                self.grad = self.grad + grad if self.grad is not None else grad

        out._backward = _backward
        out._prev = {self}
        return out

    # ====== 自动求导 ======

    def backward(self):
        if self.grad is None:
            self.grad = np.ones_like(self.data)

        topo = []
        visited = set()

        def build_topo(tensor):
            if tensor not in visited:
                visited.add(tensor)
                for child in tensor._prev:
                    build_topo(child)
                topo.append(tensor)

        build_topo(self)

        for tensor in reversed(topo):
            tensor._backward()

    def zero_grad(self):
        if self.requires_grad:
            self.grad = np.zeros_like(self.data)


# ====== 测试示例 ======

if __name__ == "__main__":
    a = Tensor([2.0], requires_grad=True)
    b = Tensor([4.0], requires_grad=True)

    c = a ** 2 + (-b) / a
    c.backward()

    print("a.grad =", a.grad)  # 应该是 2a - (-b / a²)
    print("b.grad =", b.grad)  # 应该是 -1 / a

