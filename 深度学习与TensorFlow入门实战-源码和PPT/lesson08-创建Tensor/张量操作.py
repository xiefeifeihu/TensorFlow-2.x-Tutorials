import tensorflow as tf

x = tf.constant(1.)

aa = tf.Variable(x)
print('----', aa.name)
print('----', aa.trainable)

print(tf.random.normal([2, 2], mean=1, stddev=1))  # 正态分布，均值为1，标准差为1
print(tf.random.normal([2, 2]))  # 正态分布，均值为0，标准差为1
print(tf.random.truncated_normal([2, 2], mean=0, stddev=1))  # 截断梯度消失的整体分布

print(tf.random.uniform([2, 2], minval=0, maxval=1))  # 均匀分布，0-1之间均匀采样

print(tf.random.shuffle(tf.range(10)))  # 随机打散0-9之间的数组
print(tf.one_hot(tf.range(4), depth=10))  # one_hot编码

from tensorflow.keras import layers

# Vector
print("Vector:")
net = layers.Dense(10)  # 10层网络：XW+b
net.build((4, 8))  # 输入为4行8列
print("w----", net.kernel.shape)  # 权重w为8行10列
print("b----", net.bias.shape)  # 偏置b为1行10列

print("Matrix:")
# Matrix矩阵
x = tf.random.normal([4, 784])
net = layers.Dense(10)
print("net(x)----", net(x).shape)  # net(x)为4行10列
print("w----", net.kernel.shape)  # 权重w为784行10列
print("b----", net.bias.shape)  # 偏置b为1行10列

print("4维张量:")
x = tf.random.normal((4, 32, 32, 3))  # 4张彩色照片
net = layers.Conv2D(16, kernel_size=3)
print("net(x)----", net(x).shape)  # net(x)为(4, 30, 30, 16)
print("w----", net.kernel.shape)  # 权重w为(3, 3, 3, 16)
print("b----", net.bias.shape)  # 偏置b为(16,)
