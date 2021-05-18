import io
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras import preprocessing

config = {'batch_size': 2,
          'ctx_len': 4,
          'd_model': 128,
          'dff': 512,
          'dropout': 0.1,
          'ln_epsilon': 1e-6,
          'num_head': 8,
          'num_layer': 4}
dialogue = io.open('./dataset/dialogue.txt', encoding='utf-8').read().split('\n')
for i in range(len(dialogue)):
    if len(dialogue[i]) > 0:
        dialogue[i] = '<start> ' + ' '.join(dialogue[i]) + ' <end>'
tokenizer = preprocessing.text.Tokenizer(filters='')
tokenizer.fit_on_texts(dialogue)
vocab_size = len(tokenizer.word_index) + 1
raw_data = preprocessing.sequence.pad_sequences(tokenizer.texts_to_sequences(dialogue), padding='post')
seq_len_enc = config['ctx_len'] * raw_data.shape[1]
seq_len_dec = raw_data.shape[1] - 1
train_x = []
train_y = []
for i in range(len(raw_data) - 1):
    if raw_data[i][0] == 0 or raw_data[i + 1][0] == 0:
        continue
    context = []
    j = 0
    while j < config['ctx_len']:
        if i >= j and raw_data[i - j][0] > 0:
            context.append(raw_data[i - j])
            j += 1
        else:
            break
    while j < config['ctx_len']:
        context.append(np.zeros(raw_data.shape[1]))
        j += 1
    train_x.append(np.reshape(context, seq_len_enc))
    train_y.append(raw_data[i + 1])
state = np.random.get_state()
np.random.shuffle(train_x)
np.random.set_state(state)
np.random.shuffle(train_y)


def make_batch(train_data, batch_size):
    train_data = tf.cast(train_data, tf.int64)
    return tf.reshape(train_data[:train_data.shape[0] // batch_size * batch_size],
                      (-1, batch_size, train_data.shape[1]))


train_x = make_batch(train_x, config['batch_size'])
train_y = make_batch(train_y, config['batch_size'])


def make_pe(position, d_model):
    pe = np.arange(position)[:, np.newaxis] / np.power(10000, np.arange(d_model)[np.newaxis, :] // 2 * 2 / d_model)
    pe[:, 0::2] = np.sin(pe[:, 0::2])
    pe[:, 1::2] = np.cos(pe[:, 1::2])
    return tf.cast(pe, tf.float32)


pe_enc = make_pe(seq_len_enc, config['d_model'])
pe_dec = make_pe(seq_len_dec, config['d_model'])


def make_padding_mask(seq):
    return tf.cast(tf.math.equal(seq, 0), tf.float32)[:, tf.newaxis, :]


look_ahead_mask = 1 - tf.linalg.band_part(tf.ones((seq_len_dec, seq_len_dec)), -1, 0)


def self_attention(q, k, v, mask):
    attention_weight = tf.matmul(q, k, transpose_b=True) / tf.math.sqrt(tf.cast(tf.shape(k)[-1], tf.float32))
    if mask is not None:
        attention_weight += mask * -1e9
    return tf.matmul(tf.nn.softmax(attention_weight, axis=-1), v)


class MultiHeadAttention(layers.Layer):
    def __init__(self, d_model, num_head):
        super(MultiHeadAttention, self).__init__()
        self.d_model = d_model
        self.num_head = num_head
        self.dense_q = layers.Dense(d_model)
        self.dense_k = layers.Dense(d_model)
        self.dense_v = layers.Dense(d_model)
        self.dense = layers.Dense(d_model)

    def split(self, x):
        return tf.transpose(tf.reshape(x, (tf.shape(x)[0], tf.shape(x)[1], self.num_head, -1)), perm=[0, 2, 1, 3])

    def call(self, q, k, v, mask, training):
        q = self.split(self.dense_q(q, training=training))
        k = self.split(self.dense_k(k, training=training))
        v = self.split(self.dense_v(v, training=training))
        attention_output = self_attention(q, k, v, mask)
        attention_output = tf.transpose(attention_output, perm=[0, 2, 1, 3])
        attention_output = tf.reshape(attention_output, (tf.shape(attention_output)[0], -1, self.d_model))
        return self.dense(attention_output, training=training)


class EncoderLayer(layers.Layer):
    def __init__(self, d_model, dff, dropout, ln_epsilon, num_head):
        super(EncoderLayer, self).__init__()
        self.mha = MultiHeadAttention(d_model, num_head)
        self.ffn = layers.Dense(dff, activation='relu')
        self.dense = layers.Dense(d_model)
        self.dropout1 = layers.Dropout(dropout)
        self.dropout2 = layers.Dropout(dropout)
        self.ln1 = layers.LayerNormalization(epsilon=ln_epsilon)
        self.ln2 = layers.LayerNormalization(epsilon=ln_epsilon)

    def call(self, x, mask, training):
        mha_output = self.mha(x, x, x, mask, training)
        dropout1_output = self.dropout1(mha_output, training=training)
        ln1_output = self.ln1(dropout1_output + x, training=training)
        ffn_output = self.ffn(ln1_output, training=training)
        dense_output = self.dense(ffn_output, training=training)
        dropout2_output = self.dropout2(dense_output, training=training)
        ln2_output = self.ln2(dropout2_output + ln1_output, training=training)
        return ln2_output
