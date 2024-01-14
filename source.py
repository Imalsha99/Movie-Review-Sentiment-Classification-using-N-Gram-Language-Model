#!/usr/bin/env python
# coding: utf-8

# ## Movie Review Sentiment Classification using N-Gram Language Model

# In[3]:


import string
import nltk
import re
import numpy as np
import pandas as pd

from nltk import ngrams
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

from collections import Counter

nltk.download('punkt')
nltk.download('stopwords')


# ## Step 1: Import Movie Reviews

# In[4]:


with open("Movie_Reviews.txt", "r") as file:
    movie_reviews = file.readlines()
print("Movie reviews imported successfully.")


# In[5]:


movie_reviews


# * The following code aims to split the list of movie reviews into positive and negative arrays. It defines the function 'split_reviews' that takes a list of reviews as input and separates them based on the markers 'Positive Reviews\n' and 'Negative Reviews\n'. 
# * The resulting positive and negative arrays are then printed along with a 'test_array' containing the last element of the original movie_reviews list.

# In[6]:


test_array = [movie_reviews[-1]]

def split_reviews(reviews):
    
    pos_array = []
    neg_array = []
    
    pos_index = reviews.index("Positive Reviews\n")
    neg_index = reviews.index("Negative Reviews\n")
    
    pos_array += reviews[pos_index+1: neg_index]
    neg_array += reviews[neg_index+1:]
    
    return pos_array, neg_array

movie_reviews.pop()
pos_array, neg_array = split_reviews(movie_reviews)

print(pos_array)
print(neg_array)
print(test_array)


# In[7]:


pos_df = pd.DataFrame({"review": pos_array})
neg_df = pd.DataFrame({"review": neg_array})
test_df = pd.DataFrame({"review": test_array})


# In[8]:


pos_df, neg_df, test_df


# ## Step 2: Pre-process the Text Data

# ##### Text Preprocessing Class
# ###### ----------------------------------
# 
# * The following class, **TextPreprocessing**, defines a set of methods for common text preprocessing tasks.
# * It includes functions for removing punctuation, unwanted characters, numeric digits, converting to lowercase, tokenizing, and removing English stopwords.
# * Additionally, it provides a pipeline method that applies a series of these preprocessing steps to a DataFrame column.

# In[10]:


class TextPreprocessing:
    
    def __init__(self, stopwords_list_english):
        self.stopwords_list_english=stopwords_list_english
    
    def removePunc(self, text):
        punctuationfree = "".join([i for i in text if i not in string.punctuation])
        return punctuationfree
    
    def removeUnwanted(self, text):
        text = re.sub('\n ','',text)
        text = re.sub('\n','',text)
        text = re.sub(r"^\s+","",text)
        text = re.sub(r"\s+"," ",text)
        text = re.sub(r"\u200d","",text)
        text = re.sub(r"\u200c","",text)
        
        return text
    
    def removeNum(self, text):
        remove_digits = str.maketrans('', '', string.digits)
        return text.translate(remove_digits)
    
    def lowerCase(self, text):
        return text.lower()
    
    def tokenize(self, text):
        return word_tokenize(text)
        
    def removeStopwordsEnglish(self, text):
        output= [i for i in text if i not in self.stopwords_list_english]
        return output
    
    def pipeline(self, df, column_name):
        
        df_temp = df.copy()
        df_temp[column_name] = df_temp[column_name].apply(lambda x: self.removePunc(x))
        df_temp[column_name] = df_temp[column_name].apply(lambda x: self.removeNum(x))
        df_temp[column_name] = df_temp[column_name].apply(lambda x: self.removeUnwanted(x))
        
        df_temp = df_temp[df_temp[column_name].astype(bool)].reset_index(drop=True)
        
        df_temp[column_name] = df_temp[column_name].apply(lambda x: self.lowerCase(x))
        df_temp[column_name] = df_temp[column_name].apply(lambda x: self.tokenize(x))
        df_temp[column_name] = df_temp[column_name].apply(lambda x: self.removeStopwordsEnglish(x))
                
        return df_temp
    
preprocessing = TextPreprocessing(
    stopwords_list_english=stopwords.words('english')
)


# In[11]:


pos_df = preprocessing.pipeline(pos_df, "review")
neg_df = preprocessing.pipeline(neg_df, "review")
test_df = preprocessing.pipeline(test_df, "review")


# ## Step 3: Choose N and Implement N-Gram Model (e.g., Unigram, Bigram)

# #### Using the Unigram Model

# The given dataset can be identified as a relatively small dataset with a limited number of reviews. Therefore, For a small dataset with distributed values, choosing a unigram model (N=1) can be a reasonable and practical choice. Unigrams consider each word in isolation, making them computationally less demanding and suitable for datasets where capturing complex dependencies between words may be challenging due to limited data.
# 
# When using higher-order models like Trigrams (N=3) or higher, we consider more context, which can capture richer dependencies between words. However, with a small dataset, higher-order models may suffer from the "sparsity problem." This problem arises because the model needs to estimate probabilities for all possible combinations of N-grams, and some of these combinations may not appear in the limited data.
# F for a small dataset with distributed values, starting with a Unigram model is a sensible choice due to its simplicity and the ability to handle sparsity issues. 
# 
# Therefore decided to I cohose **Unigram Model** in this scenario.

# In[12]:


#N-gram number
n = 1

# Function to tokenize and generate N-grams
def generate_ngrams(text):
    ngrams_list = list(zip(*[text[i:] for i in range(n)]))
    return ngrams_list

def process_datasets(df):
    
    temp_df = df.copy()
    
    # Create a new column with N-grams
    temp_df['ngrams'] = temp_df['review'].apply(generate_ngrams)

    # Flatten the N-grams lists and count their occurrences
    all_ngrams = [item for sublist in temp_df['ngrams'] for item in sublist]
    ngram_counts = Counter(all_ngrams)
    total_ngrams = sum(ngram_counts.values())

    # Convert the N-gram frequencies to a DataFrame
    ngram_df = pd.DataFrame(list(ngram_counts.items()), columns=['ngram', 'frequency']).sort_values(by='frequency', ascending=False)
    
    return ngram_df, total_ngrams

ngram_pos, total_pos = process_datasets(pos_df)
ngram_neg, total_neg = process_datasets(neg_df)


# In[13]:


ngram_pos


# In[23]:


ngram_neg


# ## Step 4: Calculate the N-gram probabilities for each N-gram

# In[15]:


def calculate_ngram_probabilities(df, total):
    df_records = df.to_dict('records')
    
    for i in df_records:
        print("N-gram: {} ----- Frequency: {} ----- Probability: {}".format(i['ngram'], i['frequency'], i['frequency']/total))
    
    print("==========================================")
    
calculate_ngram_probabilities(ngram_pos, total_pos)
calculate_ngram_probabilities(ngram_neg, total_neg)


# ## Step 5: Calculate N-Gram Probability for Test Review

# In[17]:


tokens = test_df['review'][0]
tokens


# In[18]:


def calculate_sentence_probability(tokens, ngram_df, total):
    
    df_records = dict(ngram_df.values)
    test_ngrams = generate_ngrams(tokens)
    
    # Calculate the probability of the sentence using ngram probabilities
    sentence_probability = 1.0  # Initialize the probability to 1.0

    for ngram in test_ngrams:
        if ngram in df_records:
            sentence_probability = sentence_probability * (df_records[ngram]/total)

    return sentence_probability
    
pos_prob = calculate_sentence_probability(tokens, ngram_pos, total_pos)
neg_prob = calculate_sentence_probability(tokens, ngram_neg, total_neg)

print("Probability of the sentence with respect to the positive dataset: ", pos_prob)
print("Probability of the sentence with respect to the negative dataset:", neg_prob)


# ## Step 6: Predict the category of the test movie review

# In[20]:


if pos_prob > neg_prob:
    print("Positive Sentiment")
elif pos_prob < neg_prob:
    print("Negative Sentiment")
else:
    print("Neutral Sentiment")


# The probability of belonging to the positive reviews dataset is approximately 7.08e-20, and the probability of belonging to the negative reviews dataset is approximately 6.28e-14. These values suggest that the test movie review is more likely associated with the negative reviews dataset, as the probability for negativity is significantly higher than that for positivity.

# ## Step 7: Concept of Perplexity and how it measures the model's performance in language modeling

# Perplexity measures how well a language model can predict the next word in a sequence, given the previous words. The lower the perplexity, the better the model is at predicting the next word. 
# 
# The perplexity of a language model on a test set is a function of the 
# probability that the language model assigns to the test
# set.
# 
# For a given test set w1, w2 · · · wN, perplexity (PP) is the 
# probability of the test set, normalized by the number of words
# 
# * The following formula can be used to calculate perplexity<b>: PPL = 2^(-log2(likelihood</b>)).
# 
# Minimizing perplexity is the same as maximizing probability.
# Perplexity is a useful metric for comparing different language models, as it is a normalized measure of cross-entropy. This means that perplexity can be compared between models with different vocabularies or trained on different datasets
