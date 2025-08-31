import torch
import numpy

x = torch.arange(0,4,dtype=torch.float32,requires_grad=True)# 允许存储x的梯度
# 也可以写成 x.requires_grad_(True) 
print(x.grad)

y = 2 * torch.dot(x, x)
print(y)

y.backward() # 使用反向传播函数对y求导
print("y = 2 * torch.dot(x, x)")
print("x.gard", x.grad) # 求导结果存储在x.grad

# 默认情况下，pytorch会累计梯度(和之前算过的梯度相加)，我们需要清除之前的值
x.grad.zero_() # _下划线代表重写，覆盖原来的值
y = x.sum()
y.backward()
print("y = x.sum()")
print("x.grad", x.grad)

x.grad.zero_()
y = x * x
y.sum().backward() #一般不对向量或者矩阵求导，要求导先转成标量
print("y = x * x, 再y.sum()")
print("x.grad", x.grad)

x.grad.zero_()
u = y.detach() # detach()代表u是常量
z = u * x
z.sum().backward()
print("z = u * x, u = x * x, u是常量")
print("x.grad", x.grad)


