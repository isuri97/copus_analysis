import json
import time
import pandas as pd

from openai import OpenAI

from model import load_data

api_key = input("Please enter your OpenAI API key")

client = OpenAI(
    api_key=api_key
)

train = pd.read_csv('tx-train.csv', sep='\t', usecols=['words','labels','sentence_id' ])
raw_sentences = train.groupby('sentence_id')['words'].apply(lambda x: ' '.join(x) + '.').reset_index()
print(raw_sentences)


train_df, eval_df, test_sentences, gold_tags, raw_sentences = load_data()

responses = []

print(f'Total no of sentences : {len(raw_sentences)}')
for sentence, gold_tags in zip(raw_sentences, gold_tags):
    # message = """
    # A metaphor is an imaginative way of describing something by referring to something else which is the same in a particular way without using the word "like" or "as".
    # Therefore, a metaphorical plant name would be a name given to a plant that contains at least one metaphorical word which would typically draw a comparison or imply a similarity between the plant and the concept or object it is being compared to, often to evoke a particular image or emotion.
    # For example, the plant name "Corn" in Chinese is "玉蜀黍" (yù shǔ shǔ), where "玉" means jade. The word "jade" does not describe an attribute of the plant itself, but a characteristic of the plant is similar to a quality of jade, thus "jade" is used metaphorically to imply the plant is as precious or as beautiful as jade. Therefore, this plant name can be determined as a metaphorical plant name.
    # Your task is to identify metaphorical flower names in the given chinese text.
    # You must identify if there are metaphoric flower names respond as a json format of yes/no (depending on there is a metaphorical flower name or not) and the list of names in a json object.
    # Example json object : {metaphoric_names_found : 'yes',metaphoric_names = ['name1','name2']}
    # Do not provide any other explanation. Just return json object with the results.
    #
    # Sentence : """ + sentence

    # you must identify if there are Holocaust domain specific named entities respond as a json format of yes/no (depending on there is a metaphorical flower name
    # or not) and the list of names in a json object.
    # Example json object : {Holocaust specific named entities : 'yes',Holocaust specific named entities = ['name1','name2']}
    # Do not provide any other explanation. Just return json object with the results.

    message = """
            you must identify if there are Holocaust domain specific named entities respond as a json format of yes/no (depending on there is a metaphorical flower name
            or not) and the list of names in a json object.
            Example json object : {Holocaust specific named entities : 'yes',Holocaust specific named entities = ['name1','name2']}
            Do not provide any other explanation. Just return json object with the results.

            Sentence : """ + sentence


    print(f'input {sentence}')
    response = client.chat.completions.create(
        messages=[
            {
                "role": "user",
                "content": message,
            }
        ],
        model="gpt-3.5-turbo-0125",
        response_format={"type": "json_object"},
        temperature=0.1,
        # max_tokens=max_tokens,
    )

    resp = response.choices[0].message.content
    object = json.loads(resp)
    responses.append(object)
    print(gold_tags)
    print(resp)
    time.sleep(0.2)

with open('chatgpt_responses_holocaust.json', "w") as json_file:
    json.dump(responses, json_file, ensure_ascii=False)
print('Done')