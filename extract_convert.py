import pandas as pd
import numpy as np
import tools
import sys
from tqdm import tqdm

sys.setrecursionlimit(100000)
"""
1. 打乱训练的输入顺序
2. 将长句按照（标点） / (座席) 分割为多个子句。
3. 计算每个子句与摘要之间的关系
4. 把关系靠前的句子找出来，作为后续label
5. 计算作为label的子句结合和最终摘要之间的重合度 /  rouge
6. 分析一下低重合度的摘要和句子是什么类型，是否需要数据清晰
"""
maxlen = 256


def text_segmentate(text, maxlen, seps, strips=None):
    """将文本按照标点符号划分为若干个短句
    # todo 考虑是否加上客户与坐席的标记
    """
    text = text.strip().strip(strips)
    if seps and len(text) > maxlen:
        pieces = text.split(seps[0])
        text, texts = '', []
        for i, p in enumerate(pieces):
            if text and p and len(text) + len(p) > maxlen - 1:
                texts.extend(text_segmentate(text, maxlen, seps[1:], strips))
                text = ''
            if i + 1 == len(pieces):
                text = text + p
            else:
                text = text + p
            #todo 有些脏数据过长了
            text = text[-512:]
        if text:
            texts.extend(text_segmentate(text, maxlen, seps[1:], strips))
        return texts
    else:
        return [text]


def text_split(text, seps, limited=True):
    """将长句按照标点分割为多个子句。
    """
    texts = text_segmentate(text, 1, seps=seps)
    if limited:
        texts = texts[-maxlen:]
    return texts


def extract_matching(texts, summaries, start_i=0, start_j=0):
    """在texts中找若干句子，使得它们连起来与summaries尽可能相似
    算法：texts和summaries都分句，然后找出summaries最长的句子，在texts
          中找与之最相似的句子作为匹配，剩下部分递归执行。
    """
    if len(texts) == 0 or len(summaries) == 0:
        return []
    i = np.argmax([len(s) for s in summaries])
    j = np.argmax([tools.compute_main_metric(t, summaries[i], 'char') for t in texts])
    lm = extract_matching(texts[:j + 1], summaries[:i], start_i, start_j)
    rm = extract_matching(
        texts[j:], summaries[i + 1:], start_i + i + 1, start_j + j
    )
    return lm + [(start_i + i, start_j + j)] + rm


def calc_coverage(summary, text):
    return len([i for i in text if i in summary]) / len(summary)


def calc_metrics(row):
    content, abstract = row['content'], row['abstract']
    texts = text_split(content, ['【坐席】', '【客户】'])
    # todo 考虑看看summary要不要以句子进行分开, abstract最后两句都是客气的语气，没有实际的意义
    # todo 用户不认可这个态度没有提取到
    summaries = text_split(abstract, u'\n。；：，')

    summaries_cc = extract_matching(texts, summaries)
    labels = sorted(set([int(i[1]) for i in summaries_cc]))

    pred_summary = ''.join([texts[i] for i in labels])
    metric = tools.compute_main_metric(pred_summary, abstract)
    couverage = calc_coverage(set(abstract), set(pred_summary))
    return texts, labels, pred_summary, metric, couverage


if __name__ == '__main__':
    log = open('run.log', 'w')
    # todo 对输入的text的中文数字变为数字
    train_df_iter = pd.read_csv('datasets/train_dataset.csv', sep='|', chunksize=1000)
    for train_df in tqdm(train_df_iter):
        try:
            train_df[['texts', 'summaries', 'pred_summary', 'metric', 'couverage']] = train_df.apply(calc_metrics, axis=1, result_type='expand')
            # train_df.to_csv('datasets/train_dataset_pre_summary.csv', sep='|', index=False, mode='a')
        except Exception as e:
            log.writelines(str(e))
    log.close()
