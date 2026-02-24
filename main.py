import os
import ssl
import re
import urllib.request
import tarfile

import pandas as pd
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

try:
    ssl._create_default_https_context = ssl._create_unverified_context
except AttributeError:
    pass

nltk.download('stopwords', quiet=True)
nltk.download('wordnet', quiet=True)

url = "https://ai.stanford.edu/~amaas/data/sentiment/aclImdb_v1.tar.gz"
filename = "aclImdb_v1.tar.gz"

if not os.path.exists("aclImdb"):
    print("Downloading dataset...")
    urllib.request.urlretrieve(url, filename)
    with tarfile.open(filename, "r:gz") as tar:
        tar.extractall(filter='data')

def load_dataset(split):
    data = []
    for label in ['pos', 'neg']:
        path = os.path.join("aclImdb", split, label)
        for file in os.listdir(path):
            with open(os.path.join(path, file), "r", encoding="utf-8") as f:
                data.append({"review": f.read(), "label": 1 if label == 'pos' else 0})
    return pd.DataFrame(data)

print("Loading data...")
train_df = load_dataset("train")
test_df = load_dataset("test")

stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

def clean_text(text):
    text = re.sub("[^a-zA-Z]", " ", text.lower())
    words = text.split()
    return " ".join([lemmatizer.lemmatize(w) for w in words if w not in stop_words])

print("Cleaning reviews...")
train_df['cleaned'] = train_df['review'].apply(clean_text)
test_df['cleaned'] = test_df['review'].apply(clean_text)

vectorizer = TfidfVectorizer(max_features=5000)
X_train = vectorizer.fit_transform(train_df['cleaned'])
X_test = vectorizer.transform(test_df['cleaned'])
y_train = train_df['label']
y_test = test_df['label']

model = MultinomialNB()
model.fit(X_train, y_train)
predictions = model.predict(X_test)

print(f"\nAccuracy: {accuracy_score(y_test, predictions) * 100:.2f}%")
print(classification_report(y_test, predictions, target_names=['Negative', 'Positive']))

os.makedirs("images", exist_ok=True)
cm = confusion_matrix(y_test, predictions)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=['Negative', 'Positive'],
            yticklabels=['Negative', 'Positive'])
plt.title('Confusion Matrix')
plt.ylabel('Actual')
plt.xlabel('Predicted')
plt.tight_layout()
plt.savefig("images/confusion_matrix.png", dpi=150)
plt.close()
