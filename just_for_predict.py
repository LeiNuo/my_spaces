import extract_convert as convert
from tools import *
from bert4keras.tokenizers import Tokenizer
from bert4keras.backend import keras, K
from bert4keras.models import build_transformer_model
from bert4keras.snippets import sequence_padding
from keras.models import Model


class GlobalAveragePooling1D(keras.layers.GlobalAveragePooling1D):
    """自定义全局池化
    """
    def call(self, inputs, mask=None):
        if mask is not None:
            mask = K.cast(mask, K.floatx())[:, :, None]
            return K.sum(inputs * mask, axis=1) / K.sum(mask, axis=1)
        else:
            return K.mean(inputs, axis=1)



tokenizer = Tokenizer(dict_path, do_lower_case=True)

# 加载bert模型，补充平均池化
encoder = build_transformer_model(
    config_path,
    checkpoint_path,
)
output = GlobalAveragePooling1D()(encoder.output)
encoder = Model(encoder.inputs, output)


def vectorize_predict(texts):
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


def predict(text, topk=3):
    # 抽取
    texts = convert.text_split(text, seps=',')
    vecs = vectorize_predict(texts)
    preds = model.predict(vecs[None])[0, :, 0]
    preds = np.where(preds > 0.2)[0]
    summary = ''.join([texts[i] for i in preds])
    # 生成
    summary = autosummary.generate(summary, topk=topk)
    # 返回
    return summary

