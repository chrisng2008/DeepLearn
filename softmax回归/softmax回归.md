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