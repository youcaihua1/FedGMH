# 论文代码

## 生成数据集
在数据集目录内使用 
python generate_MNIST.py
python generate_Cifar10.py
python generate_Cifar100.py
即可生成数据集，该脚本会自动创建目录类似于Cifar10_c2_a0.5_n100，其中c表示标签类别，a表示分布差异，n表示客户端总数。

## 运行算法
1、确保数据集已经生成
2、直接在system目录内，运行python main.py即可按照论文内的实验设置运行FedGMH算法

## 参数解释
python main.py -data Cifar100_c5_a0.5_n100 -m cnn -algo FedGMH -gr 300 -lbs 100 -lr 0.01 -ld True -ls 2 -jr 0.2 -nc 100 -tau 0.5
- -data Cifar100_c5_a0.5_n100 数据集
- -m cnn 模型
- -algo FedGMH 算法
- -gr 300 全局轮次
- -lbs 100 本地批次
- -lr 0.01 学习率
- -ld True 学习率衰减
- -ls 2 本地训练轮次
- -jr 0.2 参与率
- -nc 100 客户端总数
- -tau 0.5 超参数tau

