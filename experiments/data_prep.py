import pandas
from simpletransformers.ner import NERModel, NERArgs
import pandas as pd
from sklearn.model_selection import train_test_split
import argparse
import torch
from sklearn import metrics

import logging


# Load data
# parser = argparse.ArgumentParser(
#     description='''evaluates multiple models  ''')
# parser.add_argument('--model_name', required=False, help='model name', default="bert-base-cased")
# parser.add_argument('--model_type', required=False, help='model type', default="bert")
# parser.add_argument('--epochs', required=False, default= 3)
# arguments = parser.parse_args()
#
# test_df = pd.read_csv('test_df_wiener.csv', sep='\t', usecols=['words','labels','sentence_id'])
# train_df = pd.read_csv('train_df_wiener.csv', sep='\t', usecols=['words','labels','sentence_id'])
# val_df = pd.read_csv('val_df_wiener.csv', sep='\t', usecols=['words','labels','sentence_id'])
#
# train_df = train_df.dropna(subset=['sentence_id'])
# train_df = train_df.dropna(subset=['words'])
# train_df = train_df.dropna(subset=['labels'])
#
# test_df = test_df.dropna(subset=['sentence_id'])
# test_df = test_df.dropna(subset=['words'])
# test_df = test_df.dropna(subset=['labels'])
#
# val_df = val_df.dropna(subset=['sentence_id'])
# val_df = val_df.dropna(subset=['words'])
# val_df = val_df.dropna(subset=['labels'])
#
# unique_tags_starting_with_B = train_df['labels'].unique()
# print(unique_tags_starting_with_B)
#
# model_args = NERArgs()
# model_args.train_batch_size = 16
# model_args.eval_batch_size = 64
# model_args.overwrite_output_dir = True
# model_args.num_train_epochs = 3
# model_args.use_multiprocessing = False
# model_args.save_best_model=False
# model_args.use_multiprocessing_for_evaluation = False
# model_args.classification_report = True
# model_args.evaluate_during_training = False
# model_args.wandb_project="holo-ner"
# model_args.labels_list = ['O', 'B-DATE', 'B-PERSON', 'B-GPE', 'B-ORG', 'I-ORG', 'B-CARDINAL', 'B-LANGUAGE',
#                           'B-EVENT', 'I-DATE', 'B-NORP', 'B-TIME', 'I-TIME', 'I-GPE', 'B-ORDINAL', 'I-PERSON',
#                           'B-MILITARY',
#                           'I-MILITARY', 'I-NORP', 'B-CAMP', 'I-EVENT', 'I-CARDINAL', 'B-LAW', 'I-LAW', 'B-QUANTITY',
#                           'B-RIVER',
#                           'I-RIVER', 'B-PERCENT', 'I-PERCENT', 'B-WORK_OF_ART', 'I-QUANTITY', 'B-FAC', 'I-FAC',
#                           'I-WORK_OF_ART',
#                           'B-MONEY', 'I-MONEY', 'B-STREET', 'I-STREET', 'B-LOC', 'B-GHETTO', 'B-SEA-OCEAN',
#                           'I-SEA-OCEAN',
#                           'B-PRODUCT', 'I-CAMP', 'I-LOC', 'I-PRODUCT', 'I-GHETTO', 'B-SPOUSAL', 'I-SPOUSAL', 'B-SHIP',
#                           'I-SHIP',
#                           'B-FOREST', 'I-FOREST', 'B-GROUP', 'I-GROUP', 'B-MOUNTAIN', 'I-MOUNTAIN']
#
#
# MODEL_NAME = arguments.model_name
# MODEL_TYPE = arguments.model_type
# cuda_device = int(arguments.cuda_device)
# # MODEL_TYPE, MODEL_NAME,
# model = NERModel(
#     MODEL_TYPE, MODEL_NAME,
#     use_cuda=torch.cuda.is_available(),
#     cuda_device=cuda_device,
#     args=model_args,
# )
#
# model.train_model(train_df,eval_df= val_df)
# # model.save_model()
# print(len(test_df))
#
# results, outputs, preds_list, truths, preds = model.eval_model(test_df)
# print(results)
# preds_list = [tag for s in preds_list for tag in s]
# ll = []
# key_list = []
#
# print(truths)
# print(preds)
# # test_df['labels'] = truths
# # test_df['predicted_set'] = preds
#
# # take the label and count is it match with
# labels = ['B-SHIP', 'I-SHIP','B-GHETTO', 'I-GHETTO', 'B-STREET', 'I-STREET', 'B-MILITARY', 'I-MILITARY', 'B-DATE', 'I-DATE', 'B-PERSON', 'I-PERSON',
#           'B-GPE', 'I-GPE', 'B-TIME', 'I-TIME', 'B-EVENT', 'I-EVENT', 'B-ORG', 'I-ORG', 'B-TIME', 'I-TIME']
#
# print(truths)
# print(preds)
#
#
# classification_report_str = metrics.classification_report(truths,preds,digits=4)
#
# with open('bert-ner.txt', 'w') as output_file:
#     output_file.write(classification_report_str)
#




# inferencing with the existing language model
df = pd.read_csv('../data/ushmm.csv', sep='\t')

df_train, df_test = [x for _, x in df.groupby(df['document_id'] >= 1100)]
print(df_train.count())
print(df_test.count())

df_train = df_train.dropna(subset=['sentence_id'])
df_train = df_train.dropna(subset=['words'])
df_train = df_train.dropna(subset=['labels'])

df_test = df_test.dropna(subset=['sentence_id'])
df_test = df_test.dropna(subset=['words'])
df_test = df_test.dropna(subset=['labels'])

pattern = r'[\t\n.,?!-_]'
df_test = df_test[~df_test['words'].str.contains(pattern, regex=True)]
df_test = df_test.dropna(subset=['sentence_id', 'labels'])
df_test.reset_index(drop=True, inplace=True)

df_train = df_train[~df_train['words'].str.contains(pattern, regex=True)]
df_train = df_train.dropna(subset=['sentence_id', 'labels'])
df_train.reset_index(drop=True, inplace=True)

print(df_train.shape)
print(df_test.shape)

# reduce 'o' label from the test set
grouped_df = df_test.groupby('sentence_id').agg({'words': ' '.join, 'labels': ' '.join}).reset_index()
print(grouped_df)
df_with_tags = grouped_df[grouped_df['labels'].str.contains('B')]
df_without_tags = grouped_df[~grouped_df['labels'].str.contains('B')]
# combine 10% of all 'o' labels
sampled_rows = df_without_tags.sample(frac=0.1, random_state=42)
df_test = pd.concat([df_with_tags, sampled_rows], ignore_index=True)
print(len(df_without_tags))
print(len(df_with_tags))


df_test['words'] = df_test['words'].str.split()
df_test['labels'] = df_test['labels'].str.split()

# # reduce 'o' label from the train set
# grouped_df_train = df_train.groupby('sentence_id').agg({'words': ' '.join, 'labels': ' '.join}).reset_index()
# print(grouped_df)
# df_train_with_tags = grouped_df_train[grouped_df_train['labels'].str.contains('B')]
# df_train_without_tags = grouped_df_train[~grouped_df_train['labels'].str.contains('B')]
# # combine 10% of all 'o' labels
# sampled_rows = df_train_without_tags.sample(frac=0.1, random_state=42)
# df_train = pd.concat([df_train_with_tags, sampled_rows], ignore_index=True)
# print(len(df_train_without_tags))
# print(len(df_train_with_tags))

# df_train['words'] = df_train['words'].str.split()
# df_train['labels'] = df_train['labels'].str.split()

# Remove empty strings and specific characters (such as tabs)
# df_train['words'] = [[word.replace('\t', '').strip() for word in sublist if word.strip()] for sublist in df_train['words']]
# df_train['labels'] = [[label for label, word in zip(sublist_labels, sublist_words) if word] for sublist_labels, sublist_words in zip(df_train['labels'], df_train['words'])]

# df_train.to_csv('train_df1.csv', sep='\t', index=False)




# df_test = df_test.explode(['words', 'labels'], ignore_index=True)
#
# print(df_test)

# sentence_id_list = df_train['sentence_id']
# sentence_ids = list(set(sentence_id_list))
# sentence_ids_train, sentence_ids_val = train_test_split(sentence_ids, test_size=0.3)
#
# df_train_filtered = df_train[df_train["sentence_id"].isin(sentence_ids_train)]
# df_val = df_train[df_train["sentence_id"].isin(sentence_ids_val)]
#
# df_train_filtered.to_csv('train_df.csv', sep='\t', index=False)
# df_val.to_csv('val_df.csv', sep='\t', index=False)



# df_test.to_csv('test_df1.csv', sep='\t', index=False)
df_test.to_json('test_df1.json', orient='records')

print(f'training set size {len(df_train)}')
print(f'test set size {len(df_test)}')

#
#







