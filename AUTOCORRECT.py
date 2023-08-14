#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul 27 21:59:48 2023

@author: arinashah
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun 20 11:52:38 2023

@author: arinashah
"""
#come up with fake example
#create a test set
#can't truncate by token but only characters (kept the data size the same)
#unsure if BERT can handle the difference in singular and plural because Spellchecker module thinks that plural is a spelling error 
#it is best lemmantize: get rid of stop words 
#best to change plurals to singulars 
#best to lower case vs uppercase the text
#best to get rid of special characters and puncutation
#%% Preprocessing 
import pdb 
from spellchecker import SpellChecker
import torch
import evaluate
import pandas as pd
import numpy as np
import random
from transformers import BertTokenizer, BertForSequenceClassification, GPT2Tokenizer, GPT2ForSequenceClassification
from torch.optim import AdamW, Adam
from torch.utils.data import DataLoader
from tqdm import tqdm
from sklearn.model_selection import train_test_split, KFold, StratifiedKFold
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt 
import torch.nn.functional as F
from sklearn.utils.class_weight import compute_class_weight
import re
from torchvision import datasets, transforms
from transformers import logging
logging.set_verbosity_error()
import csv
import random
import os
def clean(text):
    # regex pattern to match HTML comments
    pattern = r"<!--(.*?)-->"
    # replace HTML comments with an empty string
    clean_string = re.sub(pattern, '', text)
    return clean_string

seed_val = 42
random.seed(seed_val)
np.random.seed(seed_val)
torch.manual_seed(seed_val)
torch.cuda.manual_seed_all(seed_val)
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
torch.use_deterministic_algorithms(True)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

MODEL = 'bert-base-uncased'
# MODEL = 'gpt2'
DATA_PATH1 = '/Users/arinashah/Documents/thesis-data-and-scripts/trainingdata/gdpr_issues_dataset1.csv'
DATA_PATH2 = '/Users/arinashah/Documents/thesis-data-and-scripts/trainingdata/gdpr_issues_dataset2.csv'
DATA_PATH4 = '/Users/arinashah/Documents/thesis-data-and-scripts/trainingdata/gdpr_issues_dataset4.csv'
DATA_PATH5 = '/Users/arinashah/Documents/thesis-data-and-scripts/trainingdata/gdpr_issues_dataset5.csv'
DATA_PATH6 = '/Users/arinashah/Documents/thesis-data-and-scripts/trainingdata/gdpr_issues_dataset6.csv'
df1 = pd.read_csv(DATA_PATH1)
df2 = pd.read_csv(DATA_PATH2)
df4 = pd.read_csv(DATA_PATH4)
df5 = pd.read_csv(DATA_PATH5)
df6 = pd.read_csv(DATA_PATH6)

# keep only certain columns
df1 = df1[['title', 'body', 'relevant', 'GDPR article']]
df2 = df2[['title', 'body', 'relevant', 'GDPR article']]
df4 = df4[['title', 'body', 'relevant', 'GDPR article']]
df4 = df4[:431]  # only keep first 431 (annotated until there)
df5 = df5[['title', 'body', 'relevant', 'GDPR article']]
df5 = df5[:1198]  # only keep first 1198 (annotated until there)
df6 = df6[['title', 'body', 'relevant', 'GDPR article']]
# print(df6.iloc[98])
df6 = df6[:98]  # only keep first 98 (annotated until there)

df = pd.concat([df1, df2, df4, df5, df6], ignore_index=True)
# only keep data labeled Y or N
df = df[df['relevant'].isin(['Y', 'N'])]
print(len(df[df['relevant'] == 'N']))
body = df.loc[:, "body"]

#file_path = '/Users/arinashah/Documents/dataset.csv'
#df.to_csv(file_path, index=False)
#%% Holdout Set
X = df.iloc[:, :]
batch_size = 32
test_size = 0.2  # 20% of the data will be allocated for testing
random_state = 42  # Set a random seed for reproducibility
X_train, X_test = train_test_split(X, test_size=test_size, random_state=random_state)
train_loader = torch.utils.data.DataLoader(X_train, batch_size=batch_size, shuffle=True)
test_loader = torch.utils.data.DataLoader(X_test, batch_size=batch_size, shuffle=False)
data = X_train.to_numpy()
#%% Checking Tokens
#BERT uses a word-peice tokenizer which breaks down words into subwords, so dogs is broken down into dog and ##s
class IssuesDataset(torch.utils.data.Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        return len(self.labels)

MODEL = 'bert-base-uncased'
tokenizer = BertTokenizer.from_pretrained(MODEL, do_lower_case=True)
# Lowercases the input when tokenizing 
#Inherits from PreTrainedTokenizer which contains methods 
# tokenizer = GPT2Tokenizer.from_pretrained(MODEL, do_lower_case=True)
# tokenizer.pad_token = tokenizer.eos_token

tokenizer.add_tokens('gdpr')
# tokenizer.add_tokens('GDPR')
tokenizer.add_tokens('https://')
tokenizer.add_tokens('checkbox')
tokenizer.add_tokens('tos')

def get_encoded_dataset(texts, labels):
    encodings = tokenizer(texts, truncation=True, padding='max_length', max_length=512)
    return IssuesDataset(encodings, labels)

# train_dataset = get_encoded_dataset(train_texts, train_labels)
# val_dataset = get_encoded_dataset(val_texts, val_labels)
# test_dataset = get_encoded_dataset(test_texts, test_labels)
data = X_train.to_numpy()
labels = {'N': 0, 'Y': 1} #parsing through to see whether the file is relevant to the GDPR or not 
data[:, 2] = np.vectorize(lambda x: labels[x])(data[:, 2])
hf_texts = []
hf_labels = []
for d in df:
    text = clean(f'{d[0]} [SEP] {d[1]}')
    hf_texts.append(text)
    hf_labels.append(d[2])
train_texts = hf_texts
train_labels = hf_labels
for i, (train_idx, val_idx) in tqdm(enumerate(skf.split(train_texts, train_labels)), total=5):
      del model
      model = BertForSequenceClassification.from_pretrained(MODEL, num_labels=2, id2label=id2label, label2id=label2id, 
                                                      output_attentions=False, output_hidden_states=False)
      model.resize_token_embeddings(len(tokenizer)) 
      model.to(device)
      model.train()

      train_texts = np.array(train_texts)
      train_labels = np.array(train_labels)
      train_dataset = get_encoded_dataset(train_texts[train_idx].tolist(), train_labels[train_idx].tolist())
train_dataset = get_encoded_dataset(train_texts[train_idx].tolist(), train_labels[train_idx].tolist())
#%% SpellChecker 
spell = SpellChecker()


# Specify the target category for spelling correction
target_category = "body"

# Filter the DataFrame based on the target category
filtered_df = df["body"]
test_df = filtered_df[:20]
# Filter the dataset based on the target category
#filtered_dataset = [item for item in df if item["body"] == target_category]

# Print the filtered dataset
def fix_spelling_errors(text):
    if isinstance(text, float):
       return str(text)
    words = text.split()
    corrected_words = []
    for word in words:
       # Regularized Expression to Exclude URLs and code snippets
       if not re.match(r"(https?://\S+)|(\b\w+\b)", word):
           corrected_word = spell.correction(word)
           # Check if the word is None after correction
           if corrected_word is not None:
               corrected_words.append(corrected_word)
           else:
               corrected_words.append(word)
       else:
           corrected_words.append(word)
    #corrected_text = ' '.join(corrected_words)     
    return ' '.join(corrected_words)  
        
#corrected_word = fix_spelling_errors(test_df)
# Apply the fix_spelling_errors function to the series
def corrected_words(text):
    if isinstance(text, float):
       return str(text)
    if pd.isna(text):
        return []

    # Skip URLs
    if re.match(r"http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+", text):
        return []

    # Skip code snippets
    if re.match(r"^[\w\s]*$", text):
        return []

    # Split the text into words
    words = text.split()

    # Identify misspelled words
    misspelled = spell.unknown(words)

    # Create a list to store corrected words
    corrected_words = []

    # Iterate over the words
    for word in words:
        # Check if the word is misspelled
        #if word in misspelled:
            #if word == "None":
                #continue
            if word is not None and word in misspelled: 
            # Correct the misspelled word
                corrected_word = spell.correction(word)
                if corrected_word != None:
            # Append the corrected word to the list
                    corrected_words.append(corrected_word)

    return corrected_words
corrected_series = test_df.apply(fix_spelling_errors)
corrected_words = test_df.apply(corrected_words)
count = 0 
for words in corrected_words:
    for  i in words: 
            count +=1      
print(count)