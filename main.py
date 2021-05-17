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
    if len(dialogue[i]) > 0:
        dialogue[i] = '<start> ' + ' '.join(dialogue[i]) + ' <end>'
tokenizer = Tokenizer(filters='')
tokenizer.fit_on_texts(dialogue)
vocab_size = len(tokenizer.word_index) + 1
raw_data = preprocessing.sequence.pad_sequences(tokenizer.texts_to_sequences(dialogue), padding='post')
seq_len_enc = ctx_len * raw_data.shape[1]
seq_len_dec = raw_data.shape[1] - 1
train_x = []
train_y = []
for i in range(len(raw_data) - 1):
    if raw_data[i][0] == 0 or raw_data[i + 1][0] == 0:
        continue
    context = []
    j = 0
    while j < ctx_len:
        if i >= j and raw_data[i - j][0] > 0:
            context.append(raw_data[i - j])
            j += 1
        else:
            break
    while j < ctx_len:
        context.append(np.zeros(raw_data.shape[1]))
        j += 1
    train_x.append(np.reshape(context, seq_len_enc))
    train_y.append(raw_data[i + 1])
state = np.random.get_state()
np.random.shuffle(train_x)
np.random.set_state(state)
np.random.shuffle(train_y)


def make_batch(train_data):
    train_data = tf.cast(train_data, tf.int64)
    return tf.reshape(train_data[:train_data.shape[0] // batch_size * batch_size],
                      (-1, batch_size, train_data.shape[1]))


train_x = make_batch(train_x)
train_y = make_batch(train_y)


def make_pe(position):
    pe = np.arange(position)[:, np.newaxis] / np.power(10000, np.arange(d_model)[np.newaxis, :] // 2 * 2 / d_model)
    pe[:, 0::2] = np.sin(pe[:, 0::2])
    pe[:, 1::2] = np.cos(pe[:, 1::2])
    return tf.cast(pe, tf.float32)


pe_enc = make_pe(seq_len_enc)
pe_dec = make_pe(seq_len_dec)


def make_padding_mask(seq):
    return tf.cast(tf.math.equal(seq, 0), tf.float32)[:, tf.newaxis, :]


look_ahead_mask = 1 - tf.linalg.band_part(tf.ones((seq_len_dec, seq_len_dec)), -1, 0)


def self_attention(q, k, v, mask):
    attention_weight = tf.matmul(q, k, transpose_b=True) / tf.math.sqrt(tf.cast(tf.shape(k)[-1], tf.float32))
    if mask is not None:
        attention_weight += mask * -1e9
    return tf.matmul(tf.nn.softmax(attention_weight, axis=-1), v)
