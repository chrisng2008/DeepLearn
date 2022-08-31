# softmax 回归
> softmax 虽叫回归，但是实际上是一个分类问题
## 回归 VS 分类 - 回归估计一个连须值
- 分类预测一个离散类别

分类
- 通常多个输出
- 输出i是预测为第i类的置信度
![图 1](assest/softmax%E5%9B%9E%E5%BD%92/IMG_20220830-231813214.png)  

回归
- 单个连续数值输出
- 自然区间R
- 跟真实值的区别作为损失
![图 2](assest/softmax%E5%9B%9E%E5%BD%92/IMG_20220830-232203266.png)  


## 从回归到多分类——均方损失
- 对类别进行一位有效编码
    $$
        y = [y_1, y_2, \dots, y_n]^T
        y_i = \begin{cases} 1 &\text{if}\ i = y \\
        0 &\text{otherwise} \end{cases}
    $$
    **实际上此类编码方式为独热编码one-hot**
- 使用均方损失训练
- 最大值最为预测
    $$
        \hat{y} = \argmax_i o_i
    $$

## 从回归到多类分类——无校验比例
- 对类别进行一位有效编码
- 最大值最为预测
    $$
        \hat{y} = \argmax_i o_i
    $$
- 需要更置信的识别正确类(大余量)
$$
o_y - o_i \geq \Delta (y, i)
$$

## 从回归到多类分类——校验比例
- 输出匹配概率(非负, 和为1)
    $$
        \hat{y} = \text{softmax}(o) \\
        \hat{y_i} = \frac{\exp{(o_i)}}{\sum_k \exp{(o_k)}}
    $$
- 概率$y$和$\hat{y}$的区别作为损失
> 李宏毅老师的课有对此做详细的讲解

## Softmax和交叉熵损失
- 交叉熵常用来衡量两个概率的区别$H(p,q)=\displaystyle \sum_i -p_i \log(q_i)$

- 将它作为损失
    $$
        l(y, \hat{y}) = -\sum_i y_i \log \hat{y_i} = -\log \hat{y_y}
    $$

- 其梯度是真实概率和预测概率的区别
    $$
        \partial_{o_i}l(y, \hat{y}) = \text{softmax}(o)_i-y_i
    $$

## 损失函数
> 损失函数：衡量真实值和预测值之间的区别
### L2 Loss
**L2 Loss**又称为均方损失, 一下为均方损失计算公式，其中$y'$为预测值
$$
    l(y, \hat{y}) = \frac{1}{2}(y-y')^2
$$


### L1 Loss
$$
l(y, y') = |y-y'|
$$

### Huber' s Robust Loss
$$
l(y, y') = \begin{cases}
  |y-y'|-\frac{1}{2}    &\text{if}\ |y - y'|>1 \\
  \frac{1}{2}(y-y')^2   &\text{otherwise} 
\end{cases}
$$
使用鲁棒损失，能够使优化比较平滑，梯度更新的也比较慢


## Softmax的简洁实现
```python
import torch
from torch import nn
from d2l import torch as d2l

batch_size = 256
train_iter, test_iter = d2l.load_data_fashion_mnist(batch_size)

# PyTorch不会隐式地调整输入的形状。因此，
# 我们在线性层前定义了展平层（flatten），来调整网络输入的形状
net = nn.Sequential(nn.Flatten(), nn.Linear(784, 10))   # 输入是784, 输出是10
# Flatten()是将一个任何形状的tensor变成我们需要的tensor， 第一个维度保留，剩下的维度全部展成向量
def init_weights(m):    # m指的是Layer
    if type(m) == nn.Linear:
        # 均值为weight, 方差为0.01
        nn.init.normal_(m.weight, std=0.01)

net.apply(init_weights);


loss = nn.CrossEntropyLoss()


trainer = torch.optim.SGD(net.parameters(), lr = 0.1)

num_epochs = 10
d2l.train_ch3(net, train_iter, test_iter, loss, num_epochs, trainer)
```

![图 1](assest/softmax%E5%9B%9E%E5%BD%92/IMG_20220831-220432775.png)  
