#combine text file to .csv file

import pandas as pd
import os

import os

# Specify the folder path containing the text files
folder_path = '/home/isuri/PycharmProjects/copus_analysis/data/ex'

text_files = [os.path.join(folder_path, file) for file in os.listdir(folder_path) if file.endswith('.txt')]

file_contents = []

for file in text_files:
    with open(file, 'r', encoding='utf-8') as f:
        content = f.read()
        file_contents.append(content)
df = pd.DataFrame({'text_content': file_contents})

df.to_csv('usholo_files.csv', index=False)


