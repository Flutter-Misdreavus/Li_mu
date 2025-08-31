import matplotlib.pyplot as plt
import random
import torch
import os
import numpy

os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

# 处理数据

# 构建人造数据集
def synthetic_data(w, b, num_examples): 
    # 生成 y = Xw + b + 噪声
    X = torch.normal(0, 1, (num_examples, len(w))) # normal(mean, std, generator) 生成正态分布随机数，mean表示正态分布的均值，std表示正态分布的标准差，generator是随机种子
    y = torch.matmul(X, w) + b # matmul适用于高维张量
    y += torch.normal(0, 0.01, y.shape)
    return X, y.reshape((-1, 1)) #reshape确保y是列向量,-1表示自动分配行数


# 将大批量拆成小批量
def data_iter(batch_size, features, labels):
    num_examples = len(features) # num_examples是整个大批量的大小
    indices = list(range(num_examples)) # 生成索引，从0到num_examples-1
    random.shuffle(indices) # 打乱索引

    for i in range(0, num_examples, batch_size):
        batch_indices = torch.tensor(indices[i : min(i + batch_size, num_examples)]) # 可能最后一份不能完整拆解
        yield features[batch_indices], labels[batch_indices]


true_w = torch.tensor([2, -3.4])
true_b = 4.2
features, labels = synthetic_data(true_w, true_b, 1000)
# features是整个数据，labels是这个数据的标签向量

print('features:', features[0], 'features.shapel', features.shape, '\nlabels:', labels[0], 'labels:', labels.shape)
plt.scatter(features[:, 1].detach().numpy(), labels.detach().numpy(), 1, c='blue') #detach()转成常量，防止进行梯度计算
#scatter(x, y, s) x, y是坐标, 要转成numpy(); s是大小
plt.show()

batch_size = 10
for X, y in data_iter(batch_size, features, labels):
    print("X:\n", X, "\ny:\n", y)


# 线性回归模型

w = torch.normal(0, 0.01, size=(2, 1), requires_grad=True)
b = torch.zeros(1, requires_grad=True) # 这里b是一个为0的标量

# 定义线性回归函数
def linreg(X, w, b):
    return torch.matmul(X, w) + b

# 定义损失函数
def squared_loss(y_hat, y):
    # 均方损失
    # y_hat是预测值
    return (y_hat - y.reshape(y_hat.shape))**2 / 2

# 定义优化算法
def sgd(params, lr, batch_size):
    # 小批量随机梯度下降
    with torch.no_grad():# 更新时不需要进行梯度计算
        for param in params:
            param -= lr * param.grad / batch_size # /batch_size得到每一个的大小,以此来更新param的值.注意传入的params中的值在这里会被修改
            param.grad.zero_()

lr = 0.03 # 学习率
num_epochs = 3 # 训练批次
net = linreg # 线性回归网络
loss = squared_loss # 损失

train_lo = loss(net(features, w, b), labels)
print(f"未训练时, loss: {float(train_lo.mean())}")

for epoch in range(num_epochs):
    for X, y in data_iter(batch_size, features, labels):
        lo = loss(net(X, w, b), y) # lo是一个(batch_size, 1)的向量
        lo.sum().backward()
        sgd([w, b], lr, batch_size)

    # 评估训练效果
    with torch.no_grad():
        train_lo = loss(net(features, w, b), labels)
        print(f"训练次数epoch: {epoch + 1}, loss: {float(train_lo.mean())}")
        print(f"w的估计误差{true_w - w.reshape(true_w.shape)}, b的估计误差{true_b - b}")

def huizhi(w, b, features, labels):
    # 绘制图像

    plt.scatter(features[:, 1].detach().numpy(), labels.detach().numpy(), 1, ) 

    slope = w[1].detach().numpy()
    intercept = b.detach().numpy()
    x = numpy.linspace(-4, 4, 2)
    y = x * slope + intercept
    plt.plot(x, y)
    plt.grid(True)
    plt.show()

huizhi(w, b, features, labels)


