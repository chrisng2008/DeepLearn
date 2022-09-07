# 使用块的网络 VGG
AlexNet最大的缺点是结构不够清晰。难以确定如何审计神经网络的框架。

## VGG Introduction
  
- AlexNet 比 LeNet更深更大来得到更好的精度

- VGG块
  - 3x3卷积(填充1)(n层，m通道) 
  - 2x2最大池化层(步幅2)

![图 2](assest/%E4%BD%BF%E7%94%A8%E5%9D%97%E7%9A%84%E7%BD%91%E7%BB%9CVGG/IMG_20220907-225513967.png)  

VGG网络主要的改进在于，将AlexNet中的卷积层，最大池化层用VGG块来表示，或者用VGG块的串联来表示
![图 3](assest/%E4%BD%BF%E7%94%A8%E5%9D%97%E7%9A%84%E7%BD%91%E7%BB%9CVGG/IMG_20220907-225813049.png)  

### VGG架构
- 多个VGG块后接全连接层
- 不同次数的重复块得到不同的架构 VGG-16, VGG-19


### 进度
- LeNet
  - 2卷积 + 池化层
  - 2 全连接层

- AlexNet
  - 更大更深
  - ReLU, Dropout, 数据增强

- VGG
  - 更大更深的AlexNet(重复的VGG块)


### 总结
- VGG使用可重复使用的卷积块拉构建深度卷积神经网络
- 不同的卷积块个数和超参数可以得到不同复杂度的变种



