class SGD:
    def __init__(self, params, lr=0.01):
        self.params = params  # 需要更新的Tensor参数列表
        self.lr = lr

    def step(self):
        for p in self.params:
            if p.grad is not None:
                p.data = p.data - self.lr * p.grad

    def zero_grad(self):
        for p in self.params:
            if p.grad is not None:
                p.grad = None
