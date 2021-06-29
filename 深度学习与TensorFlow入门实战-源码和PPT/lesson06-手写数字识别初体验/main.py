import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, optimizers, datasets

(x, y), (x_val, y_val) = datasets.mnist.load_data()  # 6万张28*28的灰度图片（0-9手写数字）
# 训练集样本
x = tf.convert_to_tensor(x, dtype=tf.float32) / 255.
# 训练集标签
y = tf.convert_to_tensor(y, dtype=tf.int32)
y = tf.one_hot(y, depth=10)
print(x.shape, y.shape)
train_dataset = tf.data.Dataset.from_tensor_slices((x, y))
train_dataset = train_dataset.batch(200)  # 一次加载200张图片

# 定义3层网络，第1层是[60K,784]*[784,512]；第2层[60K,512]*[512,256]；第3层[60K,256]*[256,10]=>[60K,10]
model = keras.Sequential([
    layers.Dense(512, activation='relu'),
    layers.Dense(256, activation='relu'),
    layers.Dense(10)])

optimizer = optimizers.SGD(learning_rate=0.001)


def train_epoch(epoch):
    # Step4.loop
    for step, (x, y) in enumerate(train_dataset):

        with tf.GradientTape() as tape:
            # [b, 28, 28] => [b, 784];b为一批200
            x = tf.reshape(x, (-1, 28 * 28))
            # Step1. compute output【正向传播，计算预测值】
            # [b, 784] => [b, 10]
            out = model(x)
            # Step2. compute loss【损失函数】
            loss = tf.reduce_sum(tf.square(out - y)) / x.shape[0]

        # Step3. optimize and update w1, w2, w3, b1, b2, b3【反向传播，计算梯度】
        grads = tape.gradient(loss, model.trainable_variables)
        # w' = w - lr * grad【更新参数】
        optimizer.apply_gradients(zip(grads, model.trainable_variables))

        if step % 100 == 0:
            print(epoch, step, 'loss:', loss.numpy())


def train():
    for epoch in range(30):
        train_epoch(epoch)


if __name__ == '__main__':
    train()
