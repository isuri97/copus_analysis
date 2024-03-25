import re

import pandas as pd

df = pd.read_csv('data/usholo_files.csv', sep=',')
df['text_content'] = df['text_content'].str.replace('\n', '')
df['text_content'] = df['text_content'].str.replace('United States Holocaust Memorial Museum', '')

df['text_content'] = df['text_content'].str.replace('UNITED STATES HOLOCAUST MEMORIAL MUSEUM', '')

def remove_content(text):
    # Define the pattern to match the content to be removed
    pattern = r'FIRST.*?Bill Benson:'
    return re.sub(pattern, '', text, flags=re.DOTALL)

# Apply the function to the 'text_content' column
df['text_content'] = df['text_content'].apply(remove_content)
df = df[df['text_content'] != '']
df = df.dropna(subset=['text_content'])
print(df.head(2))

df.to_csv('edited-us-holo.csv', index=False)