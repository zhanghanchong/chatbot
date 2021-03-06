import io
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras import losses
from tensorflow.keras import metrics
from tensorflow.keras import optimizers
from tensorflow.keras import preprocessing

config = {'adam_beta_1': 0.9,
          'adam_beta_2': 0.98,
          'adam_epsilon': 1e-9,
          'batch_size': 16,
          'ctx_len': 1,
          'd_model': 512,
          'dff': 2048,
          'dropout': 0.1,
          'epoch': 50,
          'ln_epsilon': 1e-6,
          'num_head': 8,
          'num_layer': 4,
          'warmup': 4000}
dialogue = io.open('./dataset/dialogue.txt', encoding='utf-8').read().split('\n')
for i in range(len(dialogue)):
    if len(dialogue[i]) > 0:
        dialogue[i] = '<start> ' + ' '.join(dialogue[i]) + ' <end>'
tokenizer = preprocessing.text.Tokenizer(filters='')
tokenizer.fit_on_texts(dialogue)
start_token = tokenizer.word_index['<start>']
end_token = tokenizer.word_index['<end>']
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
batch_num = train_x.shape[0]


def make_pe(position, d_model):
    pe = np.arange(position)[:, np.newaxis] / np.power(10000, np.arange(d_model)[np.newaxis, :] // 2 * 2 / d_model)
    pe[:, 0::2] = np.sin(pe[:, 0::2])
    pe[:, 1::2] = np.cos(pe[:, 1::2])
    return tf.cast(pe, tf.float32)


pe_enc = make_pe(seq_len_enc, config['d_model'])
pe_dec = make_pe(seq_len_dec, config['d_model'])


def make_look_ahead_mask(seq_len):
    return 1 - tf.linalg.band_part(tf.ones((seq_len, seq_len)), -1, 0)


def make_padding_mask(seq):
    return tf.cast(tf.math.equal(seq, 0), tf.float32)[:, tf.newaxis, tf.newaxis, :]


def self_attention(q, k, v, mask):
    attn_weight = tf.matmul(q, k, transpose_b=True) / tf.math.sqrt(tf.cast(k.shape[-1], tf.float32))
    if mask is not None:
        attn_weight += mask * -1e9
    return tf.matmul(tf.nn.softmax(attn_weight, axis=-1), v)


class MultiHeadAttention(layers.Layer):
    def __init__(self, d_model, num_head):
        super(MultiHeadAttention, self).__init__()
        self.d_model = d_model
        self.num_head = num_head
        self.depth = d_model // num_head
        self.dense_q = layers.Dense(d_model)
        self.dense_k = layers.Dense(d_model)
        self.dense_v = layers.Dense(d_model)
        self.dense = layers.Dense(d_model)

    def split(self, x):
        return tf.transpose(tf.reshape(x, (-1, x.shape[1], self.num_head, self.depth)), perm=[0, 2, 1, 3])

    def call(self, q, k, v, mask, training):
        q = self.split(self.dense_q(q, training=training))
        k = self.split(self.dense_k(k, training=training))
        v = self.split(self.dense_v(v, training=training))
        attn_out = self_attention(q, k, v, mask)
        attn_out = tf.transpose(attn_out, perm=[0, 2, 1, 3])
        attn_out = tf.reshape(attn_out, (-1, attn_out.shape[1], self.d_model))
        return self.dense(attn_out, training=training)


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
        mha_out = self.mha(x, x, x, mask, training)
        dropout1_out = self.dropout1(mha_out, training=training)
        ln1_out = self.ln1(dropout1_out + x, training=training)
        ffn_out = self.ffn(ln1_out, training=training)
        dense_out = self.dense(ffn_out, training=training)
        dropout2_out = self.dropout2(dense_out, training=training)
        ln2_out = self.ln2(dropout2_out + ln1_out, training=training)
        return ln2_out


class DecoderLayer(layers.Layer):
    def __init__(self, d_model, dff, dropout, ln_epsilon, num_head):
        super(DecoderLayer, self).__init__()
        self.mha1 = MultiHeadAttention(d_model, num_head)
        self.mha2 = MultiHeadAttention(d_model, num_head)
        self.ffn = layers.Dense(dff, activation='relu')
        self.dense = layers.Dense(d_model)
        self.dropout1 = layers.Dropout(dropout)
        self.dropout2 = layers.Dropout(dropout)
        self.dropout3 = layers.Dropout(dropout)
        self.ln1 = layers.LayerNormalization(epsilon=ln_epsilon)
        self.ln2 = layers.LayerNormalization(epsilon=ln_epsilon)
        self.ln3 = layers.LayerNormalization(epsilon=ln_epsilon)

    def call(self, x, enc_out, look_ahead_mask, padding_mask, training):
        mha1_out = self.mha1(x, x, x, look_ahead_mask, training)
        dropout1_out = self.dropout1(mha1_out, training=training)
        ln1_out = self.ln1(dropout1_out + x, training=training)
        mha2_out = self.mha2(ln1_out, enc_out, enc_out, padding_mask, training)
        dropout2_out = self.dropout2(mha2_out, training=training)
        ln2_out = self.ln2(dropout2_out + ln1_out, training=training)
        ffn_out = self.ffn(ln2_out, training=training)
        dense_out = self.dense(ffn_out, training=training)
        dropout3_out = self.dropout3(dense_out, training=training)
        ln3_out = self.ln3(dropout3_out + ln2_out, training=training)
        return ln3_out


class Encoder(layers.Layer):
    def __init__(self, d_model, dff, dropout, ln_epsilon, num_head, num_layer, pe, vocab_size_enc):
        super(Encoder, self).__init__()
        self.d_model = d_model
        self.num_layer = num_layer
        self.pe = pe
        self.embedding = layers.Embedding(vocab_size_enc, d_model)
        self.enc_layers = [EncoderLayer(d_model, dff, dropout, ln_epsilon, num_head) for _ in range(num_layer)]
        self.dropout = layers.Dropout(dropout)

    def call(self, x, mask, training):
        x = self.embedding(x, training=training) * tf.math.sqrt(tf.cast(self.d_model, tf.float32)) + self.pe
        x = self.dropout(x, training=training)
        for k in range(self.num_layer):
            x = self.enc_layers[k](x, mask, training)
        return x


class Decoder(layers.Layer):
    def __init__(self, d_model, dff, dropout, ln_epsilon, num_head, num_layer, pe, vocab_size_dec):
        super(Decoder, self).__init__()
        self.d_model = d_model
        self.num_layer = num_layer
        self.pe = pe
        self.embedding = layers.Embedding(vocab_size_dec, d_model)
        self.dec_layers = [DecoderLayer(d_model, dff, dropout, ln_epsilon, num_head) for _ in range(num_layer)]
        self.dropout = layers.Dropout(dropout)

    def call(self, x, enc_out, look_ahead_mask, padding_mask, training):
        x = self.embedding(x, training=training) * tf.math.sqrt(tf.cast(self.d_model, tf.float32)) + self.pe
        x = self.dropout(x, training=training)
        for k in range(self.num_layer):
            x = self.dec_layers[k](x, enc_out, look_ahead_mask, padding_mask, training)
        return x


class Transformer(tf.keras.Model):
    def __init__(self, d_model, dff, dropout, ln_epsilon, num_head, num_layer, vocab_size_enc, vocab_size_dec):
        super(Transformer, self).__init__()
        self.encoder = Encoder(d_model, dff, dropout, ln_epsilon, num_head, num_layer, pe_enc, vocab_size_enc)
        self.decoder = Decoder(d_model, dff, dropout, ln_epsilon, num_head, num_layer, pe_dec, vocab_size_dec)
        self.dense = layers.Dense(vocab_size_dec)

    def call(self, x_enc, x_dec, mask, look_ahead_mask, padding_mask, training):
        enc_out = self.encoder(x_enc, mask, training)
        dec_out = self.decoder(x_dec, enc_out, look_ahead_mask, padding_mask, training)
        return self.dense(dec_out, training=training)


transformer = Transformer(config['d_model'], config['dff'], config['dropout'], config['ln_epsilon'], config['num_head'],
                          config['num_layer'], vocab_size, vocab_size)


class Schedule(optimizers.schedules.LearningRateSchedule):
    def __init__(self, d_model, warmup):
        self.d_model = tf.cast(d_model, tf.float32)
        self.warmup = warmup

    def __call__(self, step):
        return tf.math.rsqrt(self.d_model) * tf.math.minimum(tf.math.rsqrt(step), step * (self.warmup ** -1.5))


optimizer = optimizers.Adam(Schedule(config['d_model'], config['warmup']), beta_1=config['adam_beta_1'],
                            beta_2=config['adam_beta_2'], epsilon=config['adam_epsilon'])
loss_object = losses.SparseCategoricalCrossentropy(from_logits=True, reduction='none')


def loss_function(real, pred):
    loss = loss_object(real, pred)
    mask = tf.cast(tf.math.logical_not(tf.math.equal(real, 0)), loss.dtype)
    return tf.reduce_sum(loss * mask) / tf.reduce_sum(mask)


def accuracy_function(real, pred):
    accuracy = tf.equal(real, tf.argmax(pred, axis=2))
    mask = tf.math.logical_not(tf.math.equal(real, 0))
    accuracy = tf.cast(tf.math.logical_and(accuracy, mask), tf.float32)
    mask = tf.cast(mask, tf.float32)
    return tf.reduce_sum(accuracy) / tf.reduce_sum(mask)


train_loss = metrics.Mean(name='train_loss')
train_accuracy = metrics.Mean(name='train_accuracy')
ckpt = tf.train.Checkpoint(transformer=transformer, optimizer=optimizer)
ckpt_manager = tf.train.CheckpointManager(ckpt, './checkpoints', max_to_keep=5)
if ckpt_manager.latest_checkpoint:
    ckpt.restore(ckpt_manager.latest_checkpoint)
train_step_signature = [
    tf.TensorSpec(shape=(None, seq_len_enc), dtype=tf.int64),
    tf.TensorSpec(shape=(None, seq_len_dec + 1), dtype=tf.int64)
]


@tf.function(input_signature=train_step_signature)
def train_step(x_enc, x_dec):
    x_dec_in = x_dec[:, :-1]
    x_dec_out = x_dec[:, 1:]
    mask = make_padding_mask(x_enc)
    combined_mask = tf.maximum(make_look_ahead_mask(seq_len_dec), make_padding_mask(x_dec_in))
    with tf.GradientTape() as tape:
        pred = transformer(x_enc, x_dec_in, mask, combined_mask, mask, True)
        loss = loss_function(x_dec_out, pred)
    grad = tape.gradient(loss, transformer.trainable_variables)
    optimizer.apply_gradients(zip(grad, transformer.trainable_variables))
    train_loss(loss)
    train_accuracy(accuracy_function(x_dec_out, pred))


for i in range(config['epoch']):
    train_loss.reset_states()
    train_accuracy.reset_states()
    for j in range(batch_num):
        train_step(train_x[j], train_y[j])
        print(f'Epoch {i + 1} Batch {j} Loss {train_loss.result():.4f} Accuracy {train_accuracy.result():.4f}')
    if (i + 1) % 5 == 0:
        ckpt_manager.save()
    print(f'Epoch {i + 1} Loss {train_loss.result():.4f} Accuracy {train_accuracy.result():.4f}')
test_x = np.zeros((1, seq_len_enc), np.int64)
while 1:
    test_x[0][seq_len_dec + 1:] = test_x[0][:-(seq_len_dec + 1)]
    sentence = input()
    sentence = ['<start> ' + ' '.join(sentence) + ' <end>']
    test_x[0][:seq_len_dec + 1] = preprocessing.sequence.pad_sequences(tokenizer.texts_to_sequences(sentence),
                                                                       maxlen=seq_len_dec + 1, padding='post')
    test_y = np.zeros((1, seq_len_dec + 1), np.int64)
    test_y[0][0] = start_token
    mask_test = make_padding_mask(test_x)
    for i in range(seq_len_dec):
        test_y_in = test_y[:, :-1]
        combined_mask_test = tf.maximum(make_look_ahead_mask(seq_len_dec), make_padding_mask(test_y_in))
        pred_test = transformer(test_x, test_y_in, mask_test, combined_mask_test, mask_test, False)
        pred_test = pred_test[0][i]
        test_y[0][i + 1] = tf.argmax(pred_test, axis=-1)
        if test_y[0][i + 1] == end_token:
            break
    for i in range(seq_len_dec):
        if test_y[0][i + 1] == end_token:
            break
        print(tokenizer.index_word[test_y[0][i + 1]], end='')
    print()
    test_x[0][seq_len_dec + 1:] = test_x[0][:-(seq_len_dec + 1)]
    test_x[0][:seq_len_dec + 1] = test_y[0]
