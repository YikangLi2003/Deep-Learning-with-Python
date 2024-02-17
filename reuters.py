from keras.datasets import reuters

WORD_DIMENSION = 10000 # 10000 different words
LABEL_DIMENSION = 46 # 46 different mutual-exclusive topics

# 将单词编码构成的列表解码为新闻英文标题字符串
WORD_MAPPING = dict([(value, key) for key, value in reuters.get_word_index().items()])
def decode_into_text(data):
    return " ".join([WORD_MAPPING.get(i - 3, '?') for i in data])

# 加载新闻标题数据
# data: 单词编码构成的列表 e.g train_data[10] = [1, 245, 273, 207, ...]
# label: 新闻主题的编码 e.g train_label[10] = 5
# len(train_data): 8982
# len(test_data): 2246
(train_data, train_labels), (test_data, test_labels) = reuters.load_data(num_words=WORD_DIMENSION)


import numpy as np

# 将数据（单词编码列表）编码为给神经网络的1D张量（向量）
# 每个单词编码列表变成10000维向量 若单词编码为n 则第n维的值设为1.0 不设则为0.0
def vectorize_sequences(sequences, dimension = WORD_DIMENSION):
    results = np.zeros((len(sequences), dimension))
    for i, sequence in enumerate(sequences):
        results[i, sequence] = 1.0
    return results

x_train = vectorize_sequences(train_data)
x_test = vectorize_sequences(test_data)

# 使用独热编码（one-hot encoding）也叫分类编码（categorical encoding）向量化标签数据
# 使用keras库内置方法：keras.utils.np_utils.to_categorical(labels) -> one_hot__labels
def to_one_hot(labels, dimension = LABEL_DIMENSION):
    results = np.zeros((len(labels), dimension))
    for i, label in enumerate(labels):
        results[i, label] = 1.0
    return results

one_hot_train_labels = to_one_hot(train_labels)
one_hot_test_labels = to_one_hot(test_labels)

"""
处理损失标签的另一种方法 直接转换为整数张量
y_train = np.array(train_labels) -> array([1, 2, 3, 4, 5, ...])
y_test = np.array(test_labels) -> array([1, 2, 3, 4, 5, ...])

对于这种形式的标签 需要选择 sparse_categorical_crossentropy
model.compile(optimizer='rmsprop',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

这个新的损失函数和 categorical_crossentropy 算法完全一样 只是接口不同
下面是两个整数与独热编码的转换例子
e.g 5 -> [0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0]
    8 -> [0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0]
"""

# 模型定义
from keras import models
from keras import layers

# Dense层堆叠的网络 每层只能访问上层的输出 丢失的信息无法被下层找回
# 16维的中间层对于46种结果的分类问题维度过小 所以本次使用64维的Dense层
model = models.Sequential()
model.add(layers.Dense(64, activation='relu', input_shape=(WORD_DIMENSION,)))
model.add(layers.Dense(64, activation='relu'))

# 最后一层输出46维的向量 使用softmax激活
# 将输出在46个不同类别上的概率分布 46个概率的总和为1
# output[i]是样本属于第i个类别的概率
model.add(layers.Dense(46, activation='softmax'))

# 编译模型‘
# 配置分类交叉熵为优化器 衡量网络输出的概率和标签的概率分布距离
# 模型的任务是将概率分布距离最小化 尽可能输出接近真实标签的值
model.compile(optimizer='rmsprop',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# 从训练集中留出1000个样本作为验证集
x_val = x_train[:1000]
partial_x_train = x_train[1000:]
y_val = one_hot_train_labels[:1000]
partial_y_train = one_hot_train_labels[1000:]

# 训练模型
history = model.fit(partial_x_train,
                    partial_y_train,
                    epochs=20,
                    batch_size=512,
                    validation_data=(x_val, y_val))

# 根据模型在训练集和验证集上的表现 过拟合在第9轮后出现
# 重新训练新网络 只进行9个轮次
model.fit(partial_x_train,
          partial_y_train,
          epochs=9,
          batch_size=512,
          validation_data=(x_val, y_val))

# 展示测试结果
loss, accuracy = model.evaluate(x_test, one_hot_test_labels)
print("Loss:", loss)
print("Accuracy:", accuracy)

# predictions.shape -> (2246, 46)
# 模型对每条样本输出46维的向量
# 每个维度是该样本为这个维度的主题的概率
# 概率的总和为1.00
# 46个概率中的最大值为模型预测的该样本的主题
predictions = model.predict(x_test)
