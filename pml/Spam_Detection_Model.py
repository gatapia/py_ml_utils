import pandas as pd
import nltk
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression

# Step 1: Download SMS Spam dataset
df = pd.read_csv('https://raw.githubusercontent.com/justmarkham/pycon-2016-tutorial/master/data/sms.tsv', sep='\t', header=None)
texts = df[1].tolist()
labels = df[0].tolist()

# Step 2: Remove Stop Words
stop_words = set(stopwords.words('english'))
def remove_stop_words(text):
    tokens = nltk.word_tokenize(text)
    filtered_text = [word for word in tokens if word.lower() not in stop_words]
    return ' '.join(filtered_text)

texts = [remove_stop_words(text) for text in texts]

# Step 3: CountVectorizer encoding
vectorizer = CountVectorizer()
X = vectorizer.fit_transform(texts)
y = labels

# Step 4: Train Logistic Regression model
lr = LogisticRegression()
lr.fit(X, y)

# Step 5: Predict whether a new message is spam or not
new_text = ["Congratulations! You have been selected as a winner. Reply WON to this number to claim your prize."]
new_text = [remove_stop_words(text) for text in new_text]
new_X = vectorizer.transform(new_text)
pred = lr.predict(new_X)

k=""

if pred[0] == 'spam':
    k="The message is spam."
else:
    k="The message is not spam."

print(k)
