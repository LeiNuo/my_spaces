import pandas as pd

from tqdm import tqdm
from tools import compute_metrics
from seq2seq_model import autosummary


def predict(text, topk=3):
    summary = autosummary.generate(text, topk=topk)
    # 返回
    return summary


if __name__ == '__main__':
    test_df = pd.read_csv('datasets/train_dataset_key_test.csv', sep='|')

    summaries = []
    metricss = []
    for text, target in tqdm(zip(test_df['source_1'], test_df['target']),desc=u'转换中'):
        summary = predict(text)
        metrics = compute_metrics(summary, target)
        summaries.append(summary)
        metricss.append(metrics['main'])
    test_df['summary'] = summaries
    test_df['metrics'] = metricss