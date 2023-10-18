import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from jupyterthemes import jtplot
jtplot.style(theme='monokai', context='notebook', ticks=True, grid=False)
pd.set_option('display.max_columns', None)
# pd.set_option('display.max_rows', None)

# ---------------------
# read csv file
reviews_df = pd.read_csv('amazon_alexa.csv')
#reviews_df = pd.read_csv('amazon_alexa.tsv', sep='\t')
reviews_df.describe()
reviews_df.info()

sns.countplot(x = reviews_df['rating'])
#plt.show()

reviews_df['verified_reviews'] = reviews_df['verified_reviews'].astype(str)

reviews_df['length'] = reviews_df['verified_reviews'].apply(len)

# plot the histogram of feedback distrbution
reviews_df['length'].plot(bins = 100, kind='hist')
sns.countplot(x = reviews_df['feedback'])

# ---------------
# plot wordcloud
# obtain positive the reviews
positive = reviews_df[reviews_df['feedback'] == 1]
# obtain negative the reviews
negative = reviews_df[reviews_df['feedback'] == 0]

# convert to list format
sentences = positive['verified_reviews'].tolist()
len(sentences)

# combine all reviews
sentences_as_one_string = " ".join(sentences)
print(sentences_as_one_string)

from wordcloud import WordCloud
plt.figure(figsize=(20,20))
plt.imshow(WordCloud().generate(sentences_as_one_string))
plt.show()

# --------------------
# --------------------

# data cleaning
import string
string.punctuation
import nltk # Natural Language tool kit
nltk.download('stopwords')
from nltk.corpus import stopwords
def message_cleaning(message):
    Test_punc_removed = [char for char in message if char not in string.punctuation]
    Test_punc_removed_join = ''.join(Test_punc_removed)
    Test_punc_removed_join_clean = [word for word in Test_punc_removed_join.split() if word.lower() not in stopwords.words('english')]
    return Test_punc_removed_join_clean

reviews_df_clean = reviews_df['verified_reviews'].apply(message_cleaning)
print(reviews_df['verified_reviews'][5])
print(reviews_df_clean[5])

#------------

from sklearn.feature_extraction.text import (CountVectorizer)

vectorizer = CountVectorizer(analyzer = message_cleaning)
reviews_countvectorizer = vectorizer.fit_transform(reviews_df['verified_reviews'])

print(vectorizer.get_feature_names_out())
print(reviews_countvectorizer.toarray())

reviews_countvectorizer.shape
reviews = pd.DataFrame(reviews_countvectorizer.toarray())
x = reviews
y = reviews_df['feedback']
print(y)

# --------------
# train and test AI/MI models
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2)

from sklearn.naive_bayes import MultinomialNB
NB_classifier = MultinomialNB()
NB_classifier.fit(X_train, y_train)

from sklearn.metrics import classification_report, confusion_matrix
y_predict_test = NB_classifier.predict(X_test)
cm = confusion_matrix(y_test, y_predict_test)
sns.heatmap(cm, annot=True)
print(classification_report(y_test, y_predict_test))

from sklearn.linear_model import LogisticRegression
model = LogisticRegression()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

cm = confusion_matrix(y_pred, y_test)
sns.heatmap(cm, annot=True)
print(classification_report(y_test, y_pred))

from sklearn.ensemble import GradientBoostingClassifier
model = GradientBoostingClassifier()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

cm = confusion_matrix(y_pred, y_test)
sns.heatmap(cm, annot=True)
plt.show()

print(classification_report(y_test, y_pred))
