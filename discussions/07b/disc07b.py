import pandas as pd
import numpy as np
import os
import re

# Q1

def duplicate_words(s):
    """
    Provide a list of all words that are duplicates in an input sentence.
    Assume that the sentences are lower case.

    :Example:
    >>> duplicate_words('let us plan for a horror movie movie this weekend')
    ['movie']
    >>> duplicate_words('I like surfing')
    []
    >>> duplicate_words('the class class is good but the tests tests are hard')
    ['class', 'tests']
    """
    pat = r'(\b\w+\b)\s+\b\1\b'
    return re.findall(pat, s)


# Q2
def laptop_details(df):
    """
    Given a df with product description - Return df with added columns of 
    processor (i3, i5), generation (9th Gen, 10th Gen), 
    storage (512 GB SSD, 1 TB HDD), display_in_inch (15.6 inch, 14 inch)

    :Example:
    >>> df = pd.read_csv('data/laptop_details.csv')
    >>> new_df = laptop_details(df)
    >>> new_df.shape
    (21, 5)
    >>> new_df['processor'].nunique()
    3
    """
    
    df['processor'] = df['laptop_description'].str.extract(r'(\bi[0-9]\b)')
    df['generation'] = df['laptop_description'].str.extract(r'(\b[0-9]{1,2}[d-t]{2}\s+Gen\b)')
    df['storage'] = df['laptop_description'].str.extract(r'(\b[0-9]{3}\s.{2}\s.{3}\b)')
    df['display_inch'] = df['laptop_description'].str.extract(r'([0-9]{2}(\.[0-9]{1,2})?\s+inch)')[0]
    
    return df


# Q3
def corpus_idf(corpus):
    """
    Given a text corpus as Series, return a dictionary with keys as words 
    and values as IDF values

    :Example:
    >>> reviews_df = pd.read_csv('data/musical_instruments_reviews.csv')
    >>> idf_dict = corpus_idf(reviews_df['reviewText'])
    >>> isinstance(idf_dict, dict)
    True
    >>> len(idf_dict.keys())
    2085
    
    """

    split = corpus.str.split()
    word_list = corpus.str.split().sum()
    word_list = list(map(lambda x : re.compile('[^A-Za-z0-9]').sub(' ',x.lower()).strip(), word_list))
    word_set = set(word_list)
    idfs = {}
    for word in word_set:
        tf_list = []
        for doc in split:
            tf_list.append(word in doc)
        idfs[word] = np.mean(tf_list)
    del idfs['']
    
    return idfs
