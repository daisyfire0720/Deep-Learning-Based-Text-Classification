# -*- coding: utf-8 -*-
"""
Created on Sun Oct 17 10:09:40 2021

@author: Daisy
"""

#%% import libraries
import re
import nltk
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.tokenize import sent_tokenize
from nltk.stem.porter import PorterStemmer
from nltk.stem.wordnet import WordNetLemmatizer
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')


#%% read the data and print the info
import pandas as pd
pd.options.display.max_rows = None
pd.options.display.max_columns = None
tqdm.pandas()
df = pd.read_csv("complaints.csv")
print(df.columns)
print(df.isnull().sum())

#%% clean the data
df = df[df["Consumer complaint narrative"].notnull()]
print(len(df))
print(df.isnull().sum())
print(df["Product"].value_counts())

#%% merge y label categories
print(len(df["Product"].unique().tolist()))

prod_dict = {'Credit reporting, credit repair services, or other personal consumer reports': 'CREDIT_SERVICE',
             'Debt collection': 'DEBT',
             'Credit card or prepaid card': 'CARD_SERVICE',
             'Mortgage': 'MORTGAGE',
             'Checking or savings account': 'ACCOUNT',
             'Money transfer, virtual currency, or money service': 'MONEY_SERVICE',
             'Vehicle loan or lease':'CONSUMER_LOAN',
             'Payday loan, title loan, or personal loan': 'CONSUMER_LOAN',
             'Student loan':'CONSUMER_LOAN'}

df = df.replace({"Product": prod_dict})
print(df["Product"].value_counts())
print(len(df["Product"].unique().tolist()))

df_use = df[['Product', 'Consumer complaint narrative']]

#%% build the nlp tokenization pipeline

# remove non-alphabetic and non-numeric words
def remove_non_alpha_num(text):
    text = re.sub(r'\d+', '', text)
    text = re.sub(r'([^\s\w]|_)+', '',text)
    text = text.strip().lower()
    return text


def word_tokenization(text):
    """input: string
        return: list of words
    """
    words = word_tokenize(text)
    return words

# Remove stop words
def remove_stopwords(words):
    """input: list of words
       return: list of words without stopwords
    """
    words = [w for w in words if w not in stopwords.words('english')]
    return words

def remove_x(words):
    """input: list of strings
         return: list of strings that doesn't contain x or xx or xxx...
    """
    new_words = []
    for i in words:
        n = len(i)
        xs = 'x' * n
        if i != xs:
            new_words.append(i)
    return new_words

def stemming_lemmatization(words):
    wnl = WordNetLemmatizer()
    porter = PorterStemmer()
    words = [wnl.lemmatize(w) if (wnl.lemmatize(w).endswith('e') or wnl.lemmatize(w).endswith('es')) else porter.stem(w) for w in words]
    return words


# text preprocessing
def text_preprocessing(text):
    text = remove_non_alpha_num(text)
    words = word_tokenization(text)
    words = remove_stopwords(words)
    words = remove_x(words)
    words = stemming_lemmatization(words)
    return words

df_use["Consumer complaint narrative"] = df_use["Consumer complaint narrative"].progress_apply(text_preprocessing)

print(df_use["Consumer complaint narrative"].head(5))

#%% encode the label and export the data
df_use = df_use.reset_index(drop = True)
df_use.columns = ["category", "input"]

from sklearn.preprocessing import LabelEncoder
encoder = LabelEncoder()
df_use["label"] = encoder.fit_transform(df_use["category"])

encoder_mapping = dict(zip(encoder.classes_, encoder.transform(encoder.classes_)))
encoder_mapping = pd.DataFrame.from_dict(encoder_mapping, orient='index', columns = ["Label"])
encoder_mapping.to_csv("encoding_label_dict_token.csv", index = True)

df_use = df_use[["input", "label"]]
train, test = train_test_split(df_use,
                               test_size = 0.2,
                               stratify = df_use["label"],
                               random_state = 42)

train = train.reset_index(drop = True)
test = test.reset_index(drop = True)

train.to_csv("complaints_train.csv", index = False)
test.to_csv("complaints_test.csv", index = False)







