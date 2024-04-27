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


df1 = pd.read_csv('../data/wiener.csv', sep='\t', quoting=csv.QUOTE_NONE, encoding='utf-8')

df_train = df1.dropna(subset=['sentence_id'])
df_train = df1.dropna(subset=['words'])
df_train = df1.dropna(subset=['labels'])

document_ids = df_train['document_id'].unique()
train_document_ids = document_ids[:100]
test_document_ids = document_ids[100:150]

train_set = df_train[df_train['document_id'].isin(train_document_ids)]
test_set = df_train[df_train['document_id'].isin(test_document_ids)]

print(f'training set size {len(train_set)}')
print(f'test set size {len(test_set)}')

# concatenate words till . and add comma
words = test_set['words']
sentence_ids = test_set['sentence_id']
labels = test_set['labels']

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

print(len(test_set))
results, outputs, preds_list, truths, preds = model.eval_model(test_set)
# print(results)
preds_list = [tag for s in preds_list for tag in s]
ll = []
key_list = []


test_set['original_test_set'] = truths
test_set['predicted_set'] = preds


def print_information(real_values, predictions):
    labels = set(real_values)
    print("Weighted Recall {}".format(recall_score(real_values, predictions, average='weighted')))
    print("Weighted Precision {}".format(precision_score(real_values, predictions, average='weighted')))
    print("Weighted F1 Score {}".format(f1_score(real_values, predictions, average='weighted')))
    print("Macro F1 Score {}".format(f1_score(real_values, predictions, average='macro')))


with open('out.txt', 'w') as f:
        with redirect_stdout(f):
            print(print_information(truths, preds))
