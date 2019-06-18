import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from string import punctuation
from wordcloud import WordCloud
from nltk import tokenize
import unidecode
import nltk
import seaborn as sns

def count_vectorizer(text):
    
    '''
    This function will be return the bag of words
    '''
    
    vectorizer = CountVectorizer(lowercase=False, max_features=50)
    bag_of_words = vectorizer.fit_transform(text)
    return bag_of_words


def train_and_test(dataframe, target):
    
    '''
    This function will be return the score of Logistic Regression
    '''
    
    X_train, X_test, y_train, y_test = train_test_split(dataframe, target, random_state=42)
    logistic_model = LogisticRegression(solver='lbfgs')
    logistic_model.fit(X_train, y_train)
    
    accuracy = logistic_model.score(X_test, y_test)
    return accuracy

def cloud_of_words(df, type_column):

    search = df[df.sentiment == type_column]
    all_words = ' '.join(list(search.text_pt))
    cloud = WordCloud(width=800, height=500, max_font_size=110, collocations=False).generate(all_words)
    return cloud


def remove_stop_words(text):
    
    text = unidecode.unidecode(text).lower()
    token_punct = tokenize.WordPunctTokenizer()
    token = token_punct.tokenize(text)
    stemmer = nltk.RSLPStemmer()

    words = nltk.corpus.stopwords.words('portuguese')
    words_without_accent = [unidecode.unidecode(item) for item in words]
    stopwords = words + words_without_accent + list(punctuation)

    without_stop_words = [stemmer.stem(item) for item in token if item not in stopwords]
    
    return " ".join(without_stop_words)

def pareto(all_words):

    token_whitespace = nltk.WhitespaceTokenizer()
    token = token_whitespace.tokenize(all_words)
    frequency = nltk.FreqDist(token)
    
    df_frequency = pd.DataFrame({'word': list(frequency.keys()), 'frequency': list(frequency.values())})
    return df_frequency
