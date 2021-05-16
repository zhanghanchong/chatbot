import io
import numpy as np
import tensorflow as tf
from tensorflow.keras import preprocessing
from tensorflow.keras.preprocessing.text import Tokenizer

batch_size = 2
ctx_len = 4
d_model = 128

dialogue = io.open('./dataset/dialogue.txt', encoding='utf-8').read().split('\n')
for i in range(len(dialogue)):
    if len(dialogue[i]) == 0:
        continue
    dialogue[i] = '<start> ' + ' '.join(dialogue[i]) + ' <end>'
tokenizer = Tokenizer(filters='')
tokenizer.fit_on_texts(dialogue)
vocab_size = len(tokenizer.word_index) + 1
raw_data = preprocessing.sequence.pad_sequences(tokenizer.texts_to_sequences(dialogue), padding='post')
seq_len = raw_data.shape[1]
train_x = []
train_y = []
for i in range(len(raw_data) - 1):
    if raw_data[i][0] == 0 or raw_data[i + 1][0] == 0:
        continue
    context = []
    flag = True
    for j in range(ctx_len):
        index = i - j
        if index >= 0 and raw_data[index][0] > 0 and flag:
            for k in range(seq_len):
                context.append(raw_data[index][k])
        else:
            flag = False
            for k in range(seq_len):
                context.append(0)
    train_x.append(context)
    train_y.append(raw_data[i + 1])
state = np.random.get_state()
np.random.shuffle(train_x)
np.random.set_state(state)
np.random.shuffle(train_y)
train_x = tf.cast(train_x, tf.int64)
train_x = tf.reshape(train_x[:train_x.shape[0] // batch_size * batch_size], (-1, batch_size, ctx_len * seq_len))
train_y = tf.cast(train_y, tf.int64)
train_y = tf.reshape(train_y[:train_y.shape[0] // batch_size * batch_size], (-1, batch_size, seq_len))
