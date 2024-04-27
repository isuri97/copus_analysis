import numpy as np
import pandas as pd
import torch.cuda
from sklearn import metrics
from sklearn.model_selection import train_test_split
import argparse
from collections import Counter
from sklearn.metrics import recall_score, precision_score, f1_score


from simpletransformers.ner import NERModel, NERArgs
import csv

from contextlib import redirect_stdout

parser = argparse.ArgumentParser(
    description='''evaluates multiple models  ''')
parser.add_argument('--model_name', required=False, help='model name', default="bert")
parser.add_argument('--model_type', required=False, help='model type', default="bert-base-cased")
parser.add_argument('--cuda_device', required=False, help='cuda device', default=0)
parser.add_argument('--train', required=False, help='train file', default='data/sample.txt')
parser.add_argument('--lr', required=False, help='Learning Rate', default=4e-5)

args = parser.parse_args()

df_train = pd.read_csv('../experiments/new_data/tx-train.csv', sep='\t', quoting=csv.QUOTE_NONE, encoding='utf-8')
df_test = pd.read_csv('../experiments/new_data/tx-train.csv', sep='\t', quoting=csv.QUOTE_NONE, encoding='utf-8')

df_test.dropna(subset=['labels'], inplace=True)

df_train = df_train.dropna(subset=['sentence_id'])
df_train = df_train.dropna(subset=['words'])
df_train = df_train.dropna(subset=['labels'])

df_test = df_test.dropna(subset=['sentence_id'])
df_test = df_test.dropna(subset=['words'])
df_test = df_test.dropna(subset=['labels'])

# df_gold_train = df_gold_train.dropna(subset=['sentence_id'])
# df_gold_train = df_gold_train.dropna(subset=['words'])
# df_gold_train = df_gold_train.dropna(subset=['labels'])

# df1 = pd.DataFrame({'document_id': df_train['document_id'], 'words': df_test['words'], 'labels': df_train['labels']})
#
# sentence_id_list = []
#
# sentence_id_seq = 0
#
# dropping_sentences = []
#
# for word in df1['words'].tolist():
#     if word == "." or word == "?" or word == "!":
#         sentence_id_list.append(sentence_id_seq)
#         sentence_id_seq += 1
#         word_count = 0
#     else:
#         sentence_id_list.append(sentence_id_seq)
#
# df1['sentence_id'] = sentence_id_list

# sentence_ids = list(set(sentence_id_list))


# sentence_ids_train, sentence_ids_test = train_test_split(sentence_ids, test_size=0.3)
# df2 = df1[df1['sentence_id'] not in dropping_sentences]

# df_train, df_test = [x for _, x in df1.groupby(df1['sentence_id'] >= 400)]

# df_train = df1[df1["sentence_id"].isin(sentence_ids_train)]
# df_test = df1[df1["sentence_id"].isin(sentence_ids_test)]  # train_test_split(df1, test_size=0.1)

print(f'training set size {len(df_train)}')
print(f'test set size {len(df_test)}')



# concatenate words till . and add comma
words = df_test['words']
sentence_ids = df_test['sentence_id']
labels = df_test['labels']

model = NERModel(
        model_type=args.model_type,
        model_name=args.model_name,
        labels=['O', 'B-DATE', 'B-PERSON', 'B-GPE', 'B-ORG', 'I-ORG', 'B-LANGUAGE',
                          'B-EVENT', 'I-DATE', 'B-TIME', 'I-TIME', 'I-GPE', 'I-PERSON',
                          'B-MILITARY', 'I-MILITARY', 'B-CAMP', 'I-EVENT', 'I-CARDINAL', 'B-LAW', 'I-LAW',
                          'B-RIVER', 'I-RIVER', 'I-QUANTITY', 'B-STREET', 'I-STREET', 'B-LOC', 'B-GHETTO',
                          'B-SEA-OCEAN',
                          'I-SEA-OCEAN', 'I-CAMP', 'I-LOC', 'I-GHETTO', 'B-SPOUSAL', 'I-SPOUSAL', 'B-SHIP',
                          'I-SHIP', 'B-FOREST', 'I-FOREST', 'B-GROUP', 'I-GROUP', 'B-MOUNTAIN', 'I-MOUNTAIN','I-FAC','I-FAC',
                'B-FAC','B-NORP','B-CARDINAL','B-PERCENT','B-QUANTITY','I-NORP','B-ORDINAL','B-WORK_OF_ART','B-MONEY','I-MONEY','I-PERCENT'
],
        use_cuda=torch.cuda.is_available(),
        args={"overwrite_output_dir": True,
              "reprocess_input_data": True,
              "num_train_epochs": 1,
              "train_batch_size": 8,
              "use_multiprocessing ": False,
              "use_multiprocessing_for_evaluation" : False
              },
    )

model.train_model(df_train)
# model.save_model()

# predictions, outputs = model.predict(sentences)

print(len(df_test))
results, outputs, preds_list, truths, preds = model.eval_model(df_test)
# print(results)
preds_list = [tag for s in preds_list for tag in s]
ll = []
key_list = []


df_test['original_test_set'] = truths
df_test['predicted_set'] = preds


def print_information(real_values, predictions):
    labels = set(real_values)
    print("Weighted Recall {}".format(recall_score(real_values, predictions, average='weighted')))
    print("Weighted Precision {}".format(precision_score(real_values, predictions, average='weighted')))
    print("Weighted F1 Score {}".format(f1_score(real_values, predictions, average='weighted')))
    print("Macro F1 Score {}".format(f1_score(real_values, predictions, average='macro')))


with open('out.txt', 'w') as f:
        with redirect_stdout(f):
            print(print_information(truths, preds))
