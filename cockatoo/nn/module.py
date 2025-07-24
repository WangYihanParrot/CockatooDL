class Module:
    def __init__(self):
        self._parameters = {}
        self._modules = {}

    def parameters(self):
        params = []
        for name, param in self._parameters.items():
            params.append(param)
        for name, module in self._modules.items():
            params.extend(module.parameters())
        return params

    def zero_grad(self):
        for param in self.parameters():
            if param.grad is not None:
                param.grad = None

    def add_parameter(self, name, param):
        self._parameters[name] = param

    def add_module(self, name, module):
        self._modules[name] = module

    def __call__(self, *args, **kwargs):
        # 约定子类重写 forward()
        return self.forward(*args, **kwargs)

    def forward(self, *args, **kwargs):
        raise NotImplementedError("Subclasses must implement forward method")