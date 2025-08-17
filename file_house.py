import os
import pandas as pd
import torch
import numpy

# 写入数据
os.makedirs(os.path.join(r'C:\Users\20810\Desktop\deep_learning\learning', 'data'), exist_ok=True)
data_file = os.path.join(r'C:\Users\20810\Desktop\deep_learning\learning', 'data', 'house.csv') #csv文件是一个用逗号分隔的表格
with open(data_file, 'w') as f:
    f.write("NumRooms,Size,Price\n")
    f.write("404,100,23000\n")
    f.write("444,NA,23065\n")
    f.write("762,NA,23600\n")
    f.write("424,500,45000\n")
    f.write("NA,140,31000\n")
    f.write("571,NA,41000\n")

# 读取数据
data = pd.read_csv(data_file)
print(data)

#处理缺失的数据，典型方法有插值和删除

#使用插值
inputs = data.iloc[:, 0:2]
outputs = data.iloc[:, 2] #因为最后一列没有NA所有不需要处理

print(inputs)
print(inputs.fillna(inputs.mean())) #fillna(num)代表将所有的NA用num填充，mean是平均值的意思
inputs = inputs.fillna(inputs.mean())

# 将数值型列转换为分类类型（object）
# 哑变量不会对数值型进行转换
#inputs['NumRooms'] = inputs['NumRooms'].astype(str)
#inputs['Size'] = inputs['Size'].astype(str)
#哑变量转换
#inputs = pd.get_dummies(inputs, dummy_na=True, dtype=int)
#print(inputs)

x = torch.tensor(inputs.values)
y = torch.tensor(outputs.values)

print(x)
print(y)