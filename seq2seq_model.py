#! -*- coding: utf-8 -*-
# 法研杯2020 司法摘要
# 生成式：正式模型
# 科学空间：https://kexue.fm

import os, json
import numpy as np
import pandas as pd
import ast
from tqdm import tqdm
import tensorflow as tf
from bert4keras.backend import keras, K, batch_gather
from bert4keras.layers import Loss
from bert4keras.models import build_transformer_model
from bert4keras.tokenizers import Tokenizer, load_vocab
from bert4keras.optimizers import Adam, extend_with_exponential_moving_average
from bert4keras.snippets import sequence_padding, open
from bert4keras.snippets import DataGenerator, AutoRegressiveDecoder
from bert4keras.snippets import longest_common_subsequence
from keras.layers import Input, Dense, Lambda
from keras.models import Model
from tools import *

# 基本参数
maxlen = 1024
batch_size = 8
epochs = 50
k_sparse = 10

user_dict_path = 'D:/work_dir/V3/natural_language/spaces/datasets/user_dict.txt'
user_dict_path_2 = 'D:/work_dir/V3/natural_language/spaces/datasets/user_dict_2.txt'
nezha_config_path = 'D:/work_dir/V3/natural_language/spaces/chinese_rbt3_L-3_H-768_A-12/bert_config_rbt3.json'
nezha_checkpoint_path = 'D:/work_dir/V3/natural_language/spaces/chinese_rbt3_L-3_H-768_A-12/bert_model.ckpt'
nezha_dict_path = 'D:/work_dir/V3/natural_language/spaces/chinese_rbt3_L-3_H-768_A-12/vocab.txt'
fold = 0


def sort_keywords(row):
    idxs = [(i, row['abstract'].find(i))for i in set(row['key_words'])]
    return [i[0] for i in sorted(idxs, key=lambda i: i[1])]


def load_data(filename):
    """加载数据
    返回：[{...}]
    """
    train_df = pd.read_csv(filename, sep='|')
    train_df['key_words'] = train_df['key_words'].apply(ast.literal_eval)
    return train_df


# 加载并精简词表
token_dict, keep_tokens = load_vocab(
    dict_path=nezha_dict_path,
    simplified=True,
    startswith=['[PAD]', '[UNK]', '[CLS]', '[SEP]'],
)
pure_tokenizer = Tokenizer(token_dict.copy(), do_lower_case=True)


def load_user_dict(filename):
    """加载用户词典
    """
    user_dict = []
    with open(filename, encoding='utf-8') as f:
        for l in f:
            w = l.split()[0]
            user_dict.append(w)
    return user_dict

# todo 为什么要加这个，不太明白
user_dict = []
for w in load_user_dict(user_dict_path) + load_user_dict(user_dict_path_2):
    if w not in token_dict:
        token_dict[w] = len(token_dict)
        user_dict.append(w)
compound_tokens = [pure_tokenizer.encode(w)[0][1:-1] for w in user_dict]

tokenizer = Tokenizer(
    token_dict,
    do_lower_case=True,
    pre_tokenize=lambda s: jieba.cut(s, HMM=False)
)


def generate_copy_labels(source, target):
    """构建copy机制对应的label
    """
    mapping = longest_common_subsequence(source, target)[1]
    source_labels = [0] * len(source)
    target_labels = [0] * len(target)
    i0, j0 = -2, -2
    for i, j in mapping:
        if i == i0 + 1 and j == j0 + 1:
            source_labels[i] = 2
            target_labels[j] = 2
        else:
            source_labels[i] = 1
            target_labels[j] = 1
        i0, j0 = i, j
    return source_labels, target_labels


def random_masking(token_ids):
    """对输入进行随机mask，增加泛化能力
    """
    rands = np.random.random(len(token_ids))
    return [
        t if r > 0.15 else np.random.choice(token_ids)
        for r, t in zip(rands, token_ids)
    ]


class data_generator(DataGenerator):
    """数据生成器
    """
    def __iter__(self, random=False):
        batch_token_ids, batch_segment_ids = [], []
        batch_output_ids, batch_labels = [], []
        for is_end, d in self.sample(random):
            i = np.random.choice(2) + 1 if random else 1
            source, target = d['source_%s' % i], d['target']
            token_ids, segment_ids = tokenizer.encode(
                source, target, maxlen=maxlen, pattern='S*ES*E'
            )
            idx = token_ids.index(tokenizer._token_end_id) + 1
            masked_token_ids = random_masking(token_ids)
            source_labels, target_labels = generate_copy_labels(
                masked_token_ids[:idx], token_ids[idx:]
            )
            labels = source_labels + target_labels[1:]
            batch_token_ids.append(masked_token_ids)
            batch_segment_ids.append(segment_ids)
            batch_output_ids.append(token_ids)
            batch_labels.append(labels)
            if len(batch_token_ids) == self.batch_size or is_end:
                batch_token_ids = sequence_padding(batch_token_ids)
                batch_segment_ids = sequence_padding(batch_segment_ids)
                batch_output_ids = sequence_padding(batch_output_ids)
                batch_labels = sequence_padding(batch_labels)
                yield [
                    batch_token_ids, batch_segment_ids, \
                    batch_output_ids, batch_labels
                ], None
                batch_token_ids, batch_segment_ids = [], []
                batch_output_ids, batch_labels = [], []


class CrossEntropy(Loss):
    """交叉熵作为loss，并mask掉输入部分
    """
    def compute_loss(self, inputs, mask=None):
        seq2seq_loss = self.compute_seq2seq_loss(inputs, mask)
        copy_loss = self.compute_copy_loss(inputs, mask)
        self.add_metric(seq2seq_loss, 'seq2seq_loss')
        self.add_metric(copy_loss, 'copy_loss')
        return seq2seq_loss + 0.5 * copy_loss

    def compute_seq2seq_loss(self, inputs, mask=None):
        y_true, y_mask, _, y_pred, _ = inputs
        y_true = y_true[:, 1:]  # 目标token_ids
        y_mask = y_mask[:, :-1] * y_mask[:, 1:]  # segment_ids，刚好指示了要预测的部分
        y_pred = y_pred[:, :-1]  # 预测序列，错开一位
        # 正loss
        pos_loss = batch_gather(y_pred, y_true[..., None])[..., 0]
        # 负loss
        y_pred = tf.nn.top_k(y_pred, k=k_sparse)[0]
        neg_loss = K.logsumexp(y_pred, axis=-1)
        # 总loss
        loss = neg_loss - pos_loss
        loss = K.sum(loss * y_mask) / K.sum(y_mask)
        return loss

    def compute_copy_loss(self, inputs, mask=None):
        _, y_mask, y_true, _, y_pred = inputs
        y_mask = K.cumsum(y_mask[:, ::-1], axis=1)[:, ::-1]
        y_mask = K.cast(K.greater(y_mask, 0.5), K.floatx())
        y_mask = y_mask[:, 1:]  # mask标记，减少一位
        y_pred = y_pred[:, :-1]  # 预测序列，错开一位
        loss = K.sparse_categorical_crossentropy(y_true, y_pred)
        loss = K.sum(loss * y_mask) / K.sum(y_mask)
        return loss


model = build_transformer_model(
    nezha_config_path,
    nezha_checkpoint_path,
    model='nezha',
    application='unilm',
    with_mlm='linear',
    keep_tokens=keep_tokens,  # 只保留keep_tokens中的字，精简原字表
    compound_tokens=compound_tokens,
)
model.summary()
output = model.get_layer('MLM-Norm').output
output = Dense(3, activation='softmax')(output)
outputs = model.outputs + [output]

# 预测用模型
model = Model(model.inputs, outputs)

# 训练用模型
y_in = Input(shape=(None,))
l_in = Input(shape=(None,))
outputs = [y_in, model.inputs[1], l_in] + outputs
outputs = CrossEntropy([3, 4])(outputs)

train_model = Model(model.inputs + [y_in, l_in], outputs)

AdamEMA = extend_with_exponential_moving_average(Adam, name='AdamEMA')
optimizer = AdamEMA(learning_rate=2e-5, ema_momentum=0.9999)
train_model.compile(optimizer=optimizer)
train_model.summary()


class AutoSummary(AutoRegressiveDecoder):
    """seq2seq解码器
    """
    def get_ngram_set(self, x, n):
        """生成ngram合集，返回结果格式是:
        {(n-1)-gram: set([n-gram的第n个字集合])}
        """
        result = {}
        for i in range(len(x) - n + 1):
            k = tuple(x[i:i + n])
            if k[:-1] not in result:
                result[k[:-1]] = set()
            result[k[:-1]].add(k[-1])
        return result

    @AutoRegressiveDecoder.wraps(default_rtype='logits', use_states=True)
    def predict(self, inputs, output_ids, states):
        token_ids, segment_ids = inputs
        token_ids = np.concatenate([token_ids, output_ids], 1)
        segment_ids = np.concatenate([segment_ids, np.ones_like(output_ids)], 1)
        prediction = self.last_token(model).predict([token_ids, segment_ids])
        # states用来缓存ngram的n值
        if states is None:
            states = [0]
        elif len(states) == 1 and len(token_ids) > 1:
            states = states * len(token_ids)
        # 根据copy标签来调整概率分布
        probas = np.zeros_like(prediction[0]) - 1000  # 最终要返回的概率分布
        for i, token_ids in enumerate(inputs[0]):
            if states[i] == 0:
                prediction[1][i, 2] *= -1  # 0不能接2
            label = prediction[1][i].argmax()  # 当前label
            if label < 2:
                states[i] = label
            else:
                states[i] += 1
            if states[i] > 0:
                ngrams = self.get_ngram_set(token_ids, states[i])
                prefix = tuple(output_ids[i, 1 - states[i]:])
                if prefix in ngrams:  # 如果确实是适合的ngram
                    candidates = ngrams[prefix]
                else:  # 没有的话就退回1gram
                    ngrams = self.get_ngram_set(token_ids, 1)
                    candidates = ngrams[tuple()]
                    states[i] = 1
                candidates = list(candidates)
                probas[i, candidates] = prediction[0][i, candidates]
            else:
                probas[i] = prediction[0][i]
            idxs = probas[i].argpartition(-k_sparse)
            probas[i, idxs[:-k_sparse]] = -1000
        return probas, states

    def generate(self, text, topk=1):
        max_c_len = maxlen - self.maxlen
        token_ids, segment_ids = tokenizer.encode(text, maxlen=max_c_len)
        output_ids = self.beam_search([token_ids, segment_ids],
                                      topk)  # 基于beam search
        return tokenizer.decode(output_ids)


autosummary = AutoSummary(
    start_id=tokenizer._token_start_id,
    end_id=tokenizer._token_end_id,
    maxlen=maxlen // 2
)


def evaluate(data, topk=1, filename=None):
    """验证集评估
    """
    if filename is not None:
        F = open(filename, 'w', encoding='utf-8')
    total_metrics = {k: 0.0 for k in metric_keys}
    for d in tqdm(data, desc=u'评估中'):
        pred_summary = autosummary.generate(d['source_1'], topk)
        metrics = compute_metrics(pred_summary, d['target'])
        for k, v in metrics.items():
            total_metrics[k] += v
        if filename is not None:
            F.write(d['target'] + '\t' + pred_summary + '\n')
            F.flush()
    if filename is not None:
        F.close()
    return {k: v / len(data) for k, v in total_metrics.items()}


class Evaluator(keras.callbacks.Callback):
    """训练回调
    """
    def on_epoch_end(self, epoch, logs=None):
        optimizer.apply_ema_weights()
        if epoch%4==1:
          temp_path = '/content/drive/MyDrive/temp_file/seq2seq_model.%s.weights' % epoch
        else:
          temp_path = 'seq2seq_model.%s.weights' % epoch
        model.save_weights(temp_path)  # 保存模型
        optimizer.reset_old_weights()


def data_split(input_data):
    # todo 可以根据原文进行排序
    input_data['source_1'] = input_data['key_words']
    input_data['source_1'] = input_data['source_1'].apply(lambda x: ''.join(x))
    input_data['source_2'] = input_data['source_1']
    input_data = input_data.rename(columns={'abstract': 'target'})
    train_df = input_data.sample(frac=0.8)[['id','source_1', 'source_2', 'target']]
    test_df = input_data[~input_data.index.isin(train_df.index)][['id', 'source_1', 'source_2', 'target']]
    return train_df, test_df


if __name__ == '__main__':
    # 加载数据
    data = load_data('datasets/train_dataset_key.csv')

    train_df, test_df = data_split(data)
    train_df.to_csv('datasets/train_dataset_key_train.csv', sep='|', index=False)
    test_df.to_csv('datasets/train_dataset_key_test.csv', sep='|', index=False)
    # callback = tf.keras.callbacks.EarlyStopping(monitor='loss', patience=3, min_delta=0.001)
    # checkpoint_filepath = '/content/drive/MyDrive/temp_file/seq2seq_model.best.weights'
    # model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
    #     filepath=checkpoint_filepath,
    #     save_weights_only=True,
    #     monitor='val_accuracy',
    #     mode='max',
    #     save_best_only=True)
    #
    # train_df = pd.read_csv('datasets/train_dataset_key_train.csv', sep='|')
    # test_df = pd.read_csv('datasets/train_dataset_key_test.csv', sep='|')
    # # valid_data = data_split(data, fold, num_folds, 'valid')
    # # 启动训练
    # evaluator = Evaluator()
    # train_generator = data_generator(train_df.to_dict('records'), batch_size)
    #
    # train_model.fit_generator(
    #     train_generator.forfit(),
    #     steps_per_epoch=len(train_generator),
    #     epochs=epochs,
    #     callbacks=[evaluator, callback, model_checkpoint_callback]
    # )
