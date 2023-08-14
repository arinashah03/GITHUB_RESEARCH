#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun 20 11:52:38 2023

@author: arinashah
"""
#come up with fake example
#create a test set
import pdb 
from autocorrect import Speller
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

def clean(text):
    # regex pattern to match HTML comments
    pattern = r"<!--(.*?)-->"
    # replace HTML comments with an empty string
    clean_string = re.sub(pattern, '', text)
    return clean_string


import random
import os

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
file_path = '/Users/arinashah/Documents/dataset.csv'
df.to_csv(file_path, index=False)
#%% 
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
# tokenizer = GPT2Tokenizer.from_pretrained(MODEL, do_lower_case=True)
# tokenizer.pad_token = tokenizer.eos_token

tokenizer.add_tokens('gdpr')
# tokenizer.add_tokens('GDPR')
tokenizer.add_tokens('https://')
tokenizer.add_tokens('checkbox')
tokenizer.add_tokens('tos')
#torch tensor data tpye 
MODEL = 'bert-base-uncased'
tokenizer = BertTokenizer.from_pretrained(MODEL, do_lower_case=True)
# Lowercases the input when tokenizing 
#Inherits from PreTrainedTokenizer which contains methods 
# tokenizer = GPT2Tokenizer.from_pretrained(MODEL, do_lower_case=True)
# tokenizer.pad_token = tokenizer.eos_token
#Segement Embedding: seperates two sentences from each other and are generally defined as 0 and 1 
#Position Embedding: gives positions to each embedding in a sequence 
tokenizer.add_tokens('gdpr')
# tokenizer.add_tokens('RGDPR')
tokenizer.add_tokens('https://')
tokenizer.add_tokens('checkbox')
tokenizer.add_tokens('tos')
#%% Test Cases 
# plurals, punctuation, code, and urls are ok - words are autocorrected 
test = ["GPDR" , "'GDPR'"] #spelled incorrectly 
tokenizer(test)
test = ["gdpr" , "'gdpr'"]
tokenizer(test)
test = ["GDPR" , "'gdpr'"]
tokenizer(test)
test = ["GDPR" , "'GDPR'"]
tokenizer(test)
test = ["GDPR" , "'GDPR.'"]
tokenizer(test)
test = ["GDPR" , "GDPR's"]
tokenizer(test)
test = ["GDPR" , "GDPRs"]
tokenizer(test)
test = ["purple" , "purples"]
tokenizer(test)
test = ["integration", "integrations"]
tokenizer(test)
test = ["complience", "compliance"]
tokenizer(test)
test = ["Block iframes until cookie accepted is not working.```$cookimize_gdpr = get_option('cookimize_gdpr'); if ( ! isset( $_COOKIE['cookieControl'] ) && 'on' === $cookimize_gdpr['cookimize_toggle_iframes'] ) { ...```https://github.com/patrickposner/easy-wp-cookie-popup/blob/f6c3311aaa74f24db8bec36591f7deb859f0a9d4/src/public/class-ci-iframe.php#L12"]
tokenizer(test)
#tokenized [gdpr, "gdpr"] to:  {'input_ids': [[101, 30522, 102], [101, 1005, 30522, 1005, 102]], ]] (so punctuation doesn't matter)
#tokenized ["integration", "integrations"] to: 'input_ids': [[101, 8346, 102], [101, 8346, 2015, 102]] (so plurals vs singulars don't matter)
#tokenized ["complience", "compliance"] to: 'input_ids': [[101, 4012, 24759, 13684, 102], [101, 12646, 102]] (spelling does matter)
#tokenized $cookimize_gdpr correctly each time, with cookimize being token = 1035 and gdpr = 30522
tokenizer["gdpr"]
test = ["cookies", "cookimize"]
tokenizer(test)
test = ["Xandr has suggested that there's a detail in the original Prebid Server GDPR enforcement that needs to be revisited.The suggestion is summarized in the 'Full Enforcement' truth table in the appendix of https://docs.google.com/document/d/1fBRaodKifv1pYsWY3ia-9K96VHUjd8kKvxZlOsozm8E/edit#Basically, we think the [IAB's document](https://github.com/InteractiveAdvertisingBureau/GDPR-Transparency-and-Consent-Framework/blob/master/TCFv2/IAB%20Tech%20Lab%20-%20Consent%20string%20and%20vendor%20list%20formats%20v2.md) is self-contradictory on this point:On one hand it says:"]
tokenizer(test)
test = ["(https://github.com/InteractiveAdvertisingBureau/GDPR-Transparency-and-Consent-Framework/blob/master/TCFv2/IAB%20Tech%20Lab%20-%20Consent%20string%20and%20vendor%20list%20formats%20v2.md)"]
tokenizer(test)
test = ["integration", "'integration."]
tokenizer(test)
test = ["integratin", "'integration."]
tokenizer(test)
#%% Using autocorrect package
#need to fix the hyperlink so that it does not autocorrect 
#autcorrect utilizes UTF-8 encoding
#the defualt of BERT is set to 30522 because that is how many words are in its CORPUS 
#but the hugging face documentation says that the token embeddings need to be resized 
#model.resize_token_embeddings(len(tokenizer))
#does input 30523 for the HTTPS token so it may work 
#lazy speller is also a class within the autocorrect 
#create a for loop of all the words in the database and then query to see which ones have a hpyer link and if the word is a hyperlink don't change it 
def splitter(words): 
    return words.split(" ") #splitting on whitespace so that words in URLS are not autocorrected
spell = Speller()
spell("Integration v 'integrattion'")
#Output: "Integration v 'integration'"
#add_words = {"GDPR" : 30522}
#new_dict = spell.nlp_data | add_words
spell.nlp_data["GDPR"] = 30522 #added GDPR so it is not corrected 
spell.nlp_data["gdpr"] = 30522
#spell.nlp_data[""]
#if "GDPR" in spell.nlp_data:
   #print("yESS")
spell("GDPR vs gdpr")
spell("https://github.com/InteractiveAdvertisingBureau/GDPR-Transparency-and-Consent-Framework/blob/master/TCFv2/IAB%20Tech%20Lab%20-%20Consent%20string%20and%20vendor%20list%20formats%20v2.md)")
#corrected to: 'https://github.com/InteractiveAdvertisingBureau/GDP-Transparency-and-Consent-Framework/blob/master/Tv2/AB%20Tech%20Lab%20-%20Consent%20string%20and%20vendor%20list%20formats%20v2.md)'
#I taken out the link changes the link 
spell = ()
#%% The method is only applied to certain parts of the text 
# attempting to only apply the speller method to certain parts of the dataset so it doesn't affect httml links 
body_text = df["body"]
body_sent = []
body_sent_string = []
for sent in body_text: 
    body_sent.append(sent)
count = 0 
count1 = 0
#%% Iterating through to prevent hpyerliks and URL from being autocorrected and altered 
body_sent_string = " ".join([str(i) for i in body_sent]) 
for sent in body_sent: 
    sent_string = str(sent)
    #sent_string = " ".join([str(i) for i in sent]) 
    body_sent_string.append(sent_string)
for i in body_sent_string: 
    for word in i: 
        if word == "On": 
            print("yes")
            break 
first_word = body_sent_string[0]
print(first_word)
first_word.split(" ")
for i in first_word: 
    print(i)
print(count)
body_sent_string.split(" ")
#%% Another method for preprocessing 
for i in body_sent: 
    " ".join([str(i) for w in i]) 
for words in body_sent_string: 
    count1+= 1
print(count1)
for body in body_sent: 
    if type(words) == float:
        words  = str(words)
    for words in body:
        #if type(words) == float:
           # words  = str(words)
        count1 +=1
   # count += 1
print(count)
print(count1)  
for word in sent: 
        print(word)
        break 
def noURL(sent): 
    for words in sent: 
        if words != "https": 
            print(words)
for sent in body_text:
    noURL(sent)
#%% Try text blob 
