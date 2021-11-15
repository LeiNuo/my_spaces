import pandas as pd
import numpy as np
from tqdm import tqdm
import ast
from bert4keras.tokenizers import Tokenizer
from bert4keras.backend import keras, K
from bert4keras.models import build_transformer_model
from bert4keras.snippets import sequence_padding
from keras.models import Model
from tools import config_path, checkpoint_path, dict_path


class GlobalAveragePooling1D(keras.layers.GlobalAveragePooling1D):
    """自定义全局池化
    """
    def call(self, inputs, mask=None):
        if mask is not None:
            mask = K.cast(mask, K.floatx())[:, :, None]
            return K.sum(inputs * mask, axis=1) / K.sum(mask, axis=1)
        else:
            return K.mean(inputs, axis=1)


def predict(texts):
    """句子列表转换为句向量
    """
    batch_token_ids, batch_segment_ids = [], []
    for text in texts:
        token_ids, segment_ids = tokenizer.encode(text, maxlen=512)
        batch_token_ids.append(token_ids)
        batch_segment_ids.append(segment_ids)
    batch_token_ids = sequence_padding(batch_token_ids)
    batch_segment_ids = sequence_padding(batch_segment_ids)
    outputs = encoder.predict([batch_token_ids, batch_segment_ids])
    return outputs


def convert(data):
    """转换所有样本
    """
    embeddings = []
    for texts in tqdm(data, desc=u'向量化'):
        outputs = predict(texts)
        embeddings.append(outputs)
    embeddings = sequence_padding(embeddings)
    return embeddings


def cc(row):
    print(row['id'])
    return ast.literal_eval(row['texts'])


if __name__ == '__main__':
    # 建立分词器
    tokenizer = Tokenizer(dict_path, do_lower_case=True)

    # 加载bert模型，补充平均池化
    encoder = build_transformer_model(
        config_path,
        checkpoint_path,
    )
    output = GlobalAveragePooling1D()(encoder.output)
    encoder = Model(encoder.inputs, output)

    train_df = pd.read_csv('datasets/train_dataset_pre_summary.csv', sep='|')
    train_df = train_df[train_df['id'] != 'id']
    texts = train_df['texts'].apply(ast.literal_eval).values.tolist()
    embeddings = convert(texts)
    np.save('datasets/train_dataset_text_embeddings', embeddings)
    train_df.to_csv('datasets/train_dataset_pre_summary.csv', sep='|', index=False)
