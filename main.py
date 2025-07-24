import cockatoo.nn as nn
import cockatoo as ck
import numpy as np
from cockatoo.tensor import Tensor

num_epochs = 500

model = nn.Linear(10, 10)
loss_fn = ck.loss.mse_loss
optimizer = ck.optim.Adam(model.parameters(), lr=0.1)

# 生成 100 个输入，维度是 10
x_data = np.random.randn(100, 10).astype(np.float32)
y_data = x_data.copy()  # 输出和输入一样

# 转成你的 Tensor
full_x = Tensor(x_data, requires_grad=False)
full_y = Tensor(y_data, requires_grad=False)

for epoch in range(num_epochs):
    output = model(full_x)
    loss = loss_fn(output, full_y)
    if epoch % 50 == 0 or epoch == num_epochs - 1:
        print(loss)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

test_x = Tensor([[0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]], requires_grad=False)
out = model(test_x)
print(test_x.shape, out.shape)
print(out)
