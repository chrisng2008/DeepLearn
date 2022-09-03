# Pytorch基础

层和块

**通过模组类来对模型进行灵活构造**

```python
# 导入相关包
import torch
from torch import nn
from torch.nn import functional as F

net = nn.Sequential(nn.Linear(20, 256), nn.ReLU(), nn.Linear(256, 10))
X = torch.rand(2, 20)
net(X)


# 任何模块都是Module的一个子类
class MLP(nn.Module):
    def __init__(self):
        super().__init__()  # 调用父类去初始化参数
        # 定义两个全连接层，作为成员变量出现
        self.hidden = nn.Linear(20, 256)
        self.out = nn.Linear(256, 10)

    def forward(self, X):
        return self.out(F.relu(self.hidden(X)))


class Myseuential(nn.Module):
    def __init__(self, *args):
        super().__init__()
        for block in args:
            self._modules[block] = block

    def forward(self, X):
        for block in self._modules.values():
            X = block(X)
        return X

net = Myseuential(nn.Linear(20, 256), nn.ReLU(), nn.Linear(256, 10))
net(X)
```

**总结**：我们在nn.Sequential()中放入我们所需要的隐藏层，按照我们的需要进行堆积。我们自己需要编写`forward`函数，此函数在我们传X进入此类时，会自动被调用。(在`super().__init__()`)里，相关的函数会自动被初始化。