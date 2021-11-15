import ast
import json
import pandas as pd
import numpy as np
from tqdm import tqdm

from extract_model import model, threshold
import pickle
from tools import FOLD

if __name__ == '__main__':
    train_df = pd.read_csv('datasets/train_dataset_pre_summary.csv', sep='|')
    train_df['texts'] = train_df['texts'].apply(ast.literal_eval)
    train_df['summaries'] = train_df['summaries'].apply(ast.literal_eval)
    data_embeddings = np.load('datasets/train_dataset_text_embeddings.npy')

    train_test = pickle.load(open('datasets/kflod.pickle', 'rb'))
    for (train_index, test_index), fold in zip(train_test, range(FOLD)):
        print("TRAIN:", train_index, "TEST:", test_index)
        X_train, X_valid = data_embeddings[train_index], data_embeddings[test_index]
        valid_data = train_df.iloc[test_index][['texts', 'pred_summary', 'summaries']].to_dict(orient='records')
        model.load_weights(f'weights/extract_model.{fold}.weights')
        y_pred = model.predict(X_valid)[:, :, 0]
        results = []
        for d, yp in tqdm(zip(valid_data, y_pred), desc=u'转换中'):
            yp = yp[:len(d['texts'])]
            yp = np.where(yp > threshold)[0]
            source_1 = ''.join([d['texts'][i] for i in yp])
            source_2 = ''.join([d['texts'][i] for i in d['summaries']])
            result = {
                'source_1': source_1,
                'source_2': source_2,
                'target': d['pred_summary'],
            }
            results.append(result)

        writer = open('datasets/seq2seq_convert', 'w')
        for i in results:
            writer.write(json.dumps(i, ensure_ascii=False) + '\n')
