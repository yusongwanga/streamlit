import streamlit as st
import pandas as pd
from io import StringIO

import pandas as pd


file = pd.read_csv('linkedin-jobs-usa.csv')


uploaded_file = st.file_uploader("Choose a file")
if uploaded_file is not None:

    # # To convert to a string based IO:
    stringio = StringIO(uploaded_file.getvalue().decode("utf-8"))
    # To read file as string:
    string_data = stringio.read()
    
    # ML code here
    # DataFrame
    import pandas as pd



    # Scikit-learn
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import LabelEncoder
    from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
    from sklearn.manifold import TSNE
    from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
    from sklearn.decomposition import LatentDirichletAllocation, NMF
    from sklearn.metrics import f1_score, accuracy_score

    # nltk
    from nltk.corpus import stopwords
    from nltk.stem import SnowballStemmer
    import nltk
    nltk.download('stopwords')
    nltk.download('punkt')
    nltk.download('averaged_perceptron_tagger')
    nltk.download('wordnet')
    from nltk.corpus import stopwords, wordnet
    from nltk.stem import WordNetLemmatizer
    from nltk.tokenize import word_tokenize
    from nltk.probability import FreqDist
    import nltk
    nltk.download('omw-1.4')



    # Utility
    import string
    import re
    import numpy as np
    import os
    from collections import Counter
    import logging
    import time
    import pickle
    import itertools
    import random
    import datetime

    # # WordCloud
    # from PIL import Image
    # from wordcloud import WordCloud, STOPWORDS, ImageColorGenerator
    # from collections import Counter, defaultdict

    # Warnings
    import warnings 
    warnings.filterwarnings('ignore')


    from sklearn.naive_bayes import MultinomialNB


    import pandas as pd
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.tree import DecisionTreeClassifier
    from sklearn.metrics import accuracy_score
    from sklearn.model_selection import train_test_split, GridSearchCV


    # Set log
    logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)



    # In[194]:


    import pandas as pd


    # ### **Step - 1.2 :** *Importing Dataset*

    # In[195]:


    df = pd.read_csv("linkedin-jobs-usa.csv")
    df1 = df


    # In[198]:


    mdf = pd.DataFrame()


    mdf['text'] = (df['title'] + ' ' + df['company'] + ' ' + df['description'] + ' ' + df['location']).copy()


    # In[200]:


    mdf['link'] = df['link'].copy()



    stop_words = set(stopwords.words("english"))
    stop_words.remove('not')
    more_stopwords = {'one', 'br', 'Po', 'th', 'sayi', 'fo', 'Unknown'}
    stop_words = stop_words.union(more_stopwords)
                    
    stemmer = SnowballStemmer("english")


    # In[205]:


    def remove_URL(text):
        url = re.compile(r'https?://\S+|www\.\S+')
        return url.sub(r'', text)


    def remove_emoji(text):
        emoji_pattern = re.compile(
            '['
            u'\U0001F600-\U0001F64F'  # emoticons
            u'\U0001F300-\U0001F5FF'  # symbols & pictographs
            u'\U0001F680-\U0001F6FF'  # transport & map symbols
            u'\U0001F1E0-\U0001F1FF'  # flags (iOS)
            u'\U00002702-\U000027B0'
            u'\U000024C2-\U0001F251'
            ']+',
            flags=re.UNICODE)
        return emoji_pattern.sub(r'', text)


    def remove_html(text):
        html = re.compile(r'^[^ ]<.*?>|&([a-z0-9]+|#[0-9]\"\'\“{1,6}|#x[0-9a-f]{1,6});[^A-Za-z0-9]+')
        return re.sub(html, '', text)


    def remove_punct(text):
        table = str.maketrans('', '', string.punctuation)
        return text.translate(table)

    def remove_quotes(text):
        quotes = re.compile(r'[^A-Za-z0-9\s]+')
        return re.sub(quotes, '', text)


    # Applying helper functions

    mdf['mod_text'] = mdf['text'].apply(lambda x: remove_URL(x))
    mdf['mod_text'] = mdf['mod_text'].apply(lambda x: remove_emoji(x))
    mdf['mod_text'] = mdf['mod_text'].apply(lambda x: remove_html(x))
    mdf['mod_text'] = mdf['mod_text'].apply(lambda x: remove_punct(x))
    mdf['mod_text'] = mdf['mod_text'].apply(lambda x: remove_quotes(x))
    mdf.head()


    # In[206]:



    mdf['tokenized'] = mdf['mod_text'].apply(word_tokenize)



    # In[208]:


    # Lower casing clean text.

    mdf['lower'] = mdf['tokenized'].apply(
        lambda x: [word.lower() for word in x])




    # In[209]:


    # Removing stopwords.

    mdf['stopwords_removed'] = mdf['lower'].apply(
        lambda x: [word for word in x if word not in stop_words])



    # In[210]:


    # Applying part of speech tags.

    mdf['pos_tags'] = mdf['stopwords_removed'].apply(nltk.tag.pos_tag)



    # In[211]:


    def get_wordnet_pos(tag):
        if tag.startswith('J'):
            return wordnet.ADJ
        elif tag.startswith('V'):
            return wordnet.VERB
        elif tag.startswith('N'):
            return wordnet.NOUN
        elif tag.startswith('R'):
            return wordnet.ADV
        else:
            return wordnet.NOUN


    mdf['wordnet_pos'] = mdf['pos_tags'].apply(
        lambda x: [(word, get_wordnet_pos(pos_tag)) for (word, pos_tag) in x])


    # In[212]:


    # Applying word lemmatizer.

    wnl = WordNetLemmatizer()

    mdf['lemmatized'] = mdf['wordnet_pos'].apply(
        lambda x: [wnl.lemmatize(word, tag) for word, tag in x])

    mdf['lemmatized'] = mdf['lemmatized'].apply(
        lambda x: [word for word in x if word not in stop_words])

    mdf['lemma_str'] = [' '.join(map(str, l)) for l in mdf['lemmatized']]



    # In[213]:




    # In[214]:


    Mdf = mdf[["lemma_str", "link"]]


    # In[215]:







    TRAIN_SIZE = 0.8





    vectorizer = TfidfVectorizer(max_features=300)

    X = vectorizer.fit_transform(Mdf['lemma_str'])


    # In[221]:


    Vecdf = pd.DataFrame(X.toarray())


    # In[222]:


    Mdf['vec'] = X.toarray().tolist()



    # In[225]:


    res = file
        


    # In[226]:


    Mdf['res'] = res


    Mdf['mod_res'] = Mdf['res'].apply(lambda x: remove_URL(x))
    Mdf['mod_res'] = Mdf['mod_res'].apply(lambda x: remove_emoji(x))
    Mdf['mod_res'] = Mdf['mod_res'].apply(lambda x: remove_html(x))
    Mdf['mod_res'] = Mdf['mod_res'].apply(lambda x: remove_punct(x))
    Mdf['mod_res'] = Mdf['mod_res'].apply(lambda x: remove_quotes(x))
    Mdf.head()


    # In[229]:


    Mdf['mod_res'] = Mdf['mod_res'].apply(word_tokenize)


    # In[230]:


    Mdf['mod_res'] = Mdf['mod_res'].apply(
        lambda x: [word.lower() for word in x])


    # In[231]:


    Mdf['mod_res'] = Mdf['mod_res'].apply(
        lambda x: [word for word in x if word not in stop_words])


    # In[232]:


    Mdf['mod_res'] = Mdf['mod_res'].apply(nltk.tag.pos_tag)


    # In[233]:


    Mdf['mod_res'] = Mdf['mod_res'].apply(
        lambda x: [(word, get_wordnet_pos(pos_tag)) for (word, pos_tag) in x])


    # In[234]:


    wnl = WordNetLemmatizer()

    Mdf['mod_res'] = Mdf['mod_res'].apply(
        lambda x: [wnl.lemmatize(word, tag) for word, tag in x])

    Mdf['mod_res'] = Mdf['mod_res'].apply(
        lambda x: [word for word in x if word not in stop_words])

    Mdf['mod_res'] = [' '.join(map(str, l)) for l in Mdf['mod_res']]


    # In[ ]:


    Mdf = mdf[["lemma_str", "link"]]


    # In[235]:


    res_matrix = vectorizer.transform(Mdf['mod_res']).toarray().tolist()


    # In[236]:


    Mdf['vec_res'] = res_matrix




    a = Mdf['vec'].map(lambda x: np.array(x))


    # In[239]:


    b = Mdf['vec_res'].map(lambda x: np.array(x))


    # In[240]:


    Mdf['score'] = (a*b).map(lambda x: sum(x))


    # In[241]:


    final = Mdf.sort_values('score', ascending=False)


    # In[168]:


    final.iloc[200]['lemma_str']


    # In[242]:


    ret = ''
    for i in final.head(10)['link'].to_numpy():
        ret = ret + i + '\n'


    # In[243]:


    # print(ret)
    st.write(ret)

