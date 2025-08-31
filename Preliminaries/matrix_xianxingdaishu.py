import torch

A = torch.arange(20).reshape(5, 4)
print("A", A)
print("A.T", A.T) #注意只有矩阵能转置

B = A.clone() #如果直接B = A,B得到的是A的地址
print("A + B", A + B)

print("")
C = torch.arange(0,60,2, dtype= torch.float32).reshape(2, 3, 5)
print("C\n", C)
print("C.sum(axis = 0)\n", C.sum(axis = 0))
print("C.sum(axis = 1)\n", C.sum(axis = 1))
print("C.mean(axis = 0)", C.mean(axis = 0))

sum_C = C.sum(axis = 0, keepdim= True) #让求和的维度不变,额外的维度长度为1
print("sum_C", sum_C)
print("C / sum_C", C / sum_C) #长度为1就可以使用广播机制

#矩阵运算
print("\n矩阵运算\n")

x = torch.arange(4, dtype= torch.float32)
y = torch.arange(6, 10, dtype= torch.float32)
D = torch.arange(12, dtype= torch.float32).reshape(3, 4)
E = torch.arange(32, dtype= torch.float32).reshape(4, 8)

print("点积 torch.dot(x, y)", torch.dot(x, y))
print("向量积 torch.my(D, x)", torch.mv(D, x)) #mv的意思是矩阵(m),向量(v)
print("矩阵乘法 torch.mm(D, E)", torch.mm(D, E))

#范数
print("L1范数 torch.abs(y).sum()", torch.abs(y).sum()) #L1范数是向量各分量的绝对值之和
print("L2范数 torch.norm(y)", torch.norm(y)) #L2范数相当于向量的模
print("弗罗贝尼乌斯范数 torch.norm(C)", torch.norm(C)) #专属于矩阵，是各个元素的平方和的平方根