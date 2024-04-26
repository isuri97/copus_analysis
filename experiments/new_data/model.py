import argparse
import json
import os

import pandas as pd
import torch
from simpletransformers.ner import NERModel
from sklearn.metrics import recall_score, precision_score, f1_score
from sklearn.model_selection import train_test_split
from sklearn import metrics
from contextlib import redirect_stdout


def print_information(predictions, real_values):
    labels = set(real_values)
    print("Weighted Recall {}".format(recall_score(real_values, predictions, average='weighted')))
    print("Weighted Precision {}".format(precision_score(real_values, predictions, average='weighted')))
    print("Weighted F1 Score {}".format(f1_score(real_values, predictions, average='weighted')))
    print("Macro F1 Score {}".format(f1_score(real_values, predictions, average='macro')))


# def format_data(dataset):
#     sentence_ids = []
#     words = []
#     labels = []
#     sent_id = 0
#     dataset = pd.DataFrame(dataset)
#     for id, sent, tags in zip(dataset['sentence_id'], dataset['words'], dataset['labels']):
#         sent_index = [sent_id] * len(tags)
#         sentence_ids.extend(sent_index)
#         words.extend(sent)
#         labels.extend(tags)
#         sent_id += 1
#     return pd.DataFrame({'sentence_id': sentence_ids, 'words': words, 'labels': labels})

def format_test_data(dataset):
    sentences_id = []
    word_list = []
    label_list = []
    dataset = pd.DataFrame(dataset)
    for sent, tags, raw_sent in zip(dataset['sentence_id'], dataset['words'], dataset['labels']):
        sentences_id.append(sent)
        word_list.append(tags)
        label_list.append(raw_sent)
    return sentences_id, word_list, label_list

def load_data():
    train = pd.read_csv('tx-train.csv', sep='\t', usecols=['words','labels','sentence_id' ])
    with open('TEST-FINAL.json', 'r') as file:
        dataset = json.load(file)

    train_df, test = train_test_split(train, test_size=0.2, shuffle=True, random_state=777)
    train_df, evaluation = train_test_split(train, test_size=0.1, shuffle=True, random_state=777)
    # train_df = format_data(train_df)
    sentence_id, test_sentences, gold_tags = format_test_data(test)
    # eval_df = format_data(evaluation)
    return train_df, evaluation, gold_tags, test_sentences

def resolve_predictions(predictions, gold_tags):
    flat_predictions = []
    flat_gold_tags = []
    count = 0
    for pred, tag_list in zip(predictions, gold_tags):
        if len(pred) == len(tag_list):
            count += 1
            flat_gold_tags.extend(tag_list)
            for result in pred:
                flat_predictions.append(list(result.values())[0])
    print(f'total predicted sentence count : {count}')
    return flat_predictions, flat_gold_tags

def run(args):
    train_df, eval_df, test_sentences, gold_tags = load_data()
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
              "train_batch_size": args.batch_size,
              "use_multiprocessing ": False,
              "use_multiprocessing_for_evaluation" : False
              },
    )

    model.train_model(train_df, eval_data=eval_df)

    # def chunk_list(lst, n):
    #     for i in range(0, len(lst), n):
    #         yield lst[i:i + n]
    #
    # sentence_chunks = chunk_list(test_sentences, 50)
    #
    # with open('predictions_results.txt', 'w') as f:
    #     for chunk in sentence_chunks:
    #         predictions, raw_outputs = model.predict(chunk, split_on_space=False)
    #         flat_predictions, flat_gold_values = resolve_predictions(predictions, gold_tags)
    #         information = print_information(flat_predictions, flat_gold_values)
    #
    #         if information is not None and isinstance(information, str):
    #             f.write(information)
    #
    #         # if not information.endswith('\n'):
    #         #     f.write('\n')
    #         # elif information is not None:
    #         #     f.write(str(information))
    #         #     f.write('\n')  # Add a newline character
    #
    #     # Add a newline between chunks for better readability
    #     f.write('\n')

    predictions, raw_outputs = model.predict(test_sentences, split_on_space=False)
    flat_predictions, flat_gold_values = resolve_predictions(predictions, gold_tags)
    print_information(flat_predictions, flat_gold_values)

    results, outputs, preds_list, truths, preds = model.eval_model(test_sentences)
    truths, preds = model.eval_model(test_sentences)
    preds_list = [tag for s in preds_list for tag in s]
    ll = []
    key_list = []

    with open('out.txt', 'w') as f:
        with redirect_stdout(f):
            print(metrics.classification_report(truths, preds, digits=4))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='''evaluates models on chinese metaphoric flower names detection''')
    parser.add_argument('--model_name', type=str, required=True, help='model_name_or_path', default="bert-base-cased")
    parser.add_argument('--model_type', type=str, required=True, help='model_type', default="bert")
    parser.add_argument('--batch_size'
                        , type=int, default=4, required=False, help='batch_size')
    args = parser.parse_args()
    run(args)

