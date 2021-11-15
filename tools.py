import jieba
from rouge import Rouge
rouge = Rouge()
# from bert4keras.snippets import open

# todo 缺少jieba的自定义词典
jieba.initialize()

FOLD = 5
metric_keys = ['main', 'rouge-1', 'rouge-2', 'rouge-l']

# bert配置
config_path = 'D:\\work_dir\\V3\\natural_language\\spaces\\chinese_roberta_wwm_ext_L-12_H-768_A-12\\bert_config.json'
checkpoint_path = 'D:\\work_dir\\V3\\natural_language\\spaces\\chinese_roberta_wwm_ext_L-12_H-768_A-12\\bert_model.ckpt'
dict_path = 'D:\\work_dir\\V3\\natural_language\\spaces\\chinese_roberta_wwm_ext_L-12_H-768_A-12\\vocab.txt'


def compute_rouge(source, target, unit='word'):
    """计算rouge-1、rouge-2、rouge-l
    """
    if unit == 'word':
        source = jieba.cut(source, HMM=False)
        target = jieba.cut(target, HMM=False)
    source, target = ' '.join(source), ' '.join(target)
    try:
        scores = rouge.get_scores(hyps=source, refs=target)
        return {
            'rouge-1': scores[0]['rouge-1']['f'],
            'rouge-2': scores[0]['rouge-2']['f'],
            'rouge-l': scores[0]['rouge-l']['f'],
        }
    except ValueError:
        return {
            'rouge-1': 0.0,
            'rouge-2': 0.0,
            'rouge-l': 0.0,
        }


def compute_metrics(source, target, unit='word'):
    """计算所有metrics
    """
    metrics = compute_rouge(source, target, unit)
    metrics['main'] = (
        metrics['rouge-1'] * 0.2 + metrics['rouge-2'] * 0.4 +
        metrics['rouge-l'] * 0.4
    )
    return metrics


def compute_main_metric(source, target, unit='word'):
    """计算主要metric
    """
    # print(f'source:{source}\t', f'target:{target}')
    return compute_metrics(source, target, unit)['main']


if __name__ == '__main__':
    import numpy as np
    from sklearn.model_selection import KFold
    kf = KFold(n_splits=FOLD, shuffle=True)
    cc = kf.split(np.empty((25000, 1)))
    cc = [(train_index, test_index) for (train_index, test_index) in cc]
    import pickle
    pickle.dump(cc, open('datasets/kflod.pickle', 'wb'))
    a = 0