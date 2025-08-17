import torch

print("hello")

x = torch.arange(12)
print(x) #x是一个张量
print("x.shape", x.shape) #x的形状
print("x.numel", x.numel()) #x中元素是数字
print("id(x)", id(x)) #id相当于地址

print(x.reshape(3, 4)) #改变x的形状，改成三行四列

print(torch.zeros(2,4,5,6)) #创建全为0的张量
print(torch.ones(3,4,6)) #创建全为1的张量

y = torch.tensor([[ 0,  1,  2,  3],[ 4,  5,  6,  7],[ 8,  9, 10, 11]]) #直接创建
print("y.shape", y.shape)
print("y[-1]", y[-1])
print("y[1:3]", y[1:3])

print("\n运算\n")
x = torch.tensor([1.0, 2, 5, 3])
y = torch.tensor([3, 2, 3, 4])

#注意以下不是矩阵的运算，是对同一位置的每个数字进行运算
print(x + y)
print(x - y)
print(x * y)
print(x / y)
print(x**y)
print("x求和", x.sum()) #对x中所有元素进行求和，返回一个标量

print("\n合并\n")

x = torch.arange(12, dtype=torch.float32).reshape(3,4)
y = torch.arange(3, 15, dtype=torch.float32).reshape(3, 4)

print(torch.cat((x, y), dim= 0)) #通过dim维对齐来合并
print(torch.cat((x, y), dim= 1))

print("\n")

