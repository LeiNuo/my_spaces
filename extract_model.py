import json
import ast
import pandas as pd
import numpy as np
from tqdm import tqdm
from bert4keras.backend import keras, K
from bert4keras.layers import LayerNormalization
from bert4keras.optimizers import Adam
from keras.layers import *
from keras.models import Model
from tools import metric_keys, compute_metrics, FOLD
import pickle


class ResidualGatedConv1D(Layer):
    """门控卷积
    """
    def __init__(self, filters, kernel_size, dilation_rate=1, **kwargs):
        super(ResidualGatedConv1D, self).__init__(**kwargs)
        self.filters = filters
        self.kernel_size = kernel_size
        self.dilation_rate = dilation_rate
        self.supports_masking = True

    def build(self, input_shape):
        super(ResidualGatedConv1D, self).build(input_shape)
        self.conv1d = Conv1D(
            filters=self.filters * 2,
            kernel_size=self.kernel_size,
            dilation_rate=self.dilation_rate,
            padding='same',
        )
        self.layernorm = LayerNormalization()

        if self.filters != input_shape[-1]:
            self.dense = Dense(self.filters, use_bias=False)

        self.alpha = self.add_weight(
            name='alpha', shape=[1], initializer='zeros'
        )

    def call(self, inputs, mask=None):
        if mask is not None:
            mask = K.cast(mask, K.floatx())
            inputs = inputs * mask[:, :, None]

        outputs = self.conv1d(inputs)
        gate = K.sigmoid(outputs[..., self.filters:])
        outputs = outputs[..., :self.filters] * gate
        outputs = self.layernorm(outputs)

        if hasattr(self, 'dense'):
            inputs = self.dense(inputs)

        return inputs + self.alpha * outputs

    def compute_output_shape(self, input_shape):
        shape = self.conv1d.compute_output_shape(input_shape)
        return (shape[0], shape[1], shape[2] // 2)


def evaluate(data, data_x, threshold=0.2):
    """验证集评估
    """
    y_pred = model.predict(data_x)[:, :, 0]
    total_metrics = {k: 0.0 for k in metric_keys}
    for d, yp in tqdm(zip(data, y_pred), desc=u'评估中'):
        yp = yp[:len(d['texts'])]
        yp = np.where(yp > threshold)[0]
        pred_summary = ''.join([d['texts'][i] for i in yp])
        # todo 评价的时候出现过长文本会被阶段为512，目前还不晓得有啥影响
        metrics = compute_metrics(pred_summary[-512:], d['pred_summary'], 'char')
        for k, v in metrics.items():
            total_metrics[k] += v
    return {k: v / len(data) for k, v in total_metrics.items()}


class Evaluator(keras.callbacks.Callback):
    """训练回调
    """
    def __init__(self):
        self.best_metric = 0.0

    def on_epoch_end(self, epoch, logs=None):
        # todo threshold 应该会变
        metrics = evaluate(valid_data, X_valid, threshold + 0.1)
        if metrics['main'] >= self.best_metric:  # 保存最优
            self.best_metric = metrics['main']
            model.save_weights('weights/extract_model.%s.weights' % fold)
        metrics['best'] = self.best_metric
        print(metrics)


def create_model():
    x_in = Input(shape=(None, input_size))
    x = x_in

    x = Masking()(x)
    x = Dropout(0.1)(x)
    x = Dense(hidden_size, use_bias=False)(x)
    x = Dropout(0.1)(x)
    x = ResidualGatedConv1D(hidden_size, 3, dilation_rate=1)(x)
    x = Dropout(0.1)(x)
    x = ResidualGatedConv1D(hidden_size, 3, dilation_rate=2)(x)
    x = Dropout(0.1)(x)
    x = ResidualGatedConv1D(hidden_size, 3, dilation_rate=4)(x)
    x = Dropout(0.1)(x)
    x = ResidualGatedConv1D(hidden_size, 3, dilation_rate=8)(x)
    x = Dropout(0.1)(x)
    x = ResidualGatedConv1D(hidden_size, 3, dilation_rate=1)(x)
    x = Dropout(0.1)(x)
    x = ResidualGatedConv1D(hidden_size, 3, dilation_rate=1)(x)
    x = Dropout(0.1)(x)
    x = Dense(1, activation='sigmoid')(x)

    model = Model(x_in, x)
    model.compile(
        loss='binary_crossentropy', optimizer=Adam(), metrics=['accuracy']
    )
    model.summary()
    return model


input_size = 768
hidden_size = 384
epochs = 20
batch_size = 64
threshold = 0.2


model = create_model()

if __name__ == '__main__':
    train_idx = pd.read_csv('datasets/train_dataset_key_train.csv', sep='|')['id']
    whole_df = pd.read_csv('datasets/train_dataset_key.csv', sep='|')
    whole_df['key_words'] = whole_df['key_words'].apply(ast.literal_eval)
    train_df = whole_df[whole_df['id'].isin(train_idx)]
    test_df = whole_df[~whole_df['id'].isin(train_idx)]
    test_index = test_df['id'].values

    train_word_dict = pickle.load(open('datasets/train_word_dict.pickle', 'rb'))
    vocab_size = len(train_word_dict) + 1
    max_value = vocab_size-1
    text_length = 16

    data_x = np.load('datasets/train_dataset_text_embeddings.npy')
    data_y = np.zeros([whole_df.shape[0], vocab_size, text_length])
    for line_index, one_key_words in enumerate(whole_df['key_words']):
        for step, text in enumerate(one_key_words):
            data_y[line_index, train_word_dict.get(text, max_value), step] = 1

    # train_test = pickle.load(open('datasets/kflod.pickle', 'rb'))
    # for (train_index, test_index), fold in zip(train_test, range(FOLD)):
    X_train, X_valid = data_x[train_idx], data_x[test_index]
    y_train, y_valid = data_y[train_idx], data_y[test_index]
    valid_data = test_df[['texts', 'pred_summary']].to_dict(orient='records')
    # 启动训练
    evaluator = Evaluator()

    model.fit(
        X_train,
        y_train,
        epochs=epochs,
        batch_size=batch_size,
        callbacks=[evaluator]
    )