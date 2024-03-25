import argparse
import pandas as pd
import torch
from simpletransformers.ner import NERModel
from sklearn.metrics import recall_score, precision_score, f1_score
from sklearn.model_selection import train_test_split

def print_information(predictions, real_values):
    labels = set(real_values)
    print("Weighted Recall {}".format(recall_score(real_values, predictions, average='weighted')))
    print("Weighted Precision {}".format(precision_score(real_values, predictions, average='weighted')))
    print("Weighted F1 Score {}".format(f1_score(real_values, predictions, average='weighted')))
    print("Macro F1 Score {}".format(f1_score(real_values, predictions, average='macro')))

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
    # train, test = train_test_split(dataset, test_size=0.2, shuffle=True, random_state=777)
    train = pd.read_csv('train_df1.csv', sep='\t', usecols=['words','labels','sentence_id' ])
    test = pd.read_csv('test_df.csv', sep='\t', usecols=['words','labels','sentence_id'])
    train_df, evaluation = train_test_split(train, test_size=0.1, shuffle=True, random_state=777)
    test_sentences, gold_tags, raw_sentences = format_test_data(test)
    return train_df, evaluation, test_sentences, gold_tags, raw_sentences

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
    train_df, eval_df, test_sentences, gold_tags, raw_sentences = load_data()
    model = NERModel(
        model_type=args.model_type,
        model_name=args.model_name,
        labels=['B', 'I', 'O'],
        use_cuda=torch.cuda.is_available(),
        args={"overwrite_output_dir": True,
              "reprocess_input_data": True,
              "num_train_epochs": 3,
              "train_batch_size": args.batch_size,
              "model_args.use_multiprocessing ": False,
              "model_args.use_multiprocessing_for_evaluation" : False
              },
    )

    model.train_model(train_df, eval_data=eval_df)
    predictions, raw_outputs = model.predict(test_sentences, split_on_space=False)
    flat_predictions, flat_gold_values = resolve_predictions(predictions, gold_tags)
    print_information(flat_predictions, flat_gold_values)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='''evaluates models on chinese metaphoric flower names detection''')
    parser.add_argument('--model_name', type=str, required=True, help='model_name_or_path', default="bert-base-cased")
    parser.add_argument('--model_type', type=str, required=True, help='model_type', default="bert")
    parser.add_argument('--batch_size'
                        , type=int, default=8, required=False, help='batch_size')
    args = parser.parse_args()
    run(args)