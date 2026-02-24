import os
import urllib.request
import tarfile
import pandas as pd
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import re
import ssl
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report

# ==========================================
# MAC SSL FIX & NLTK SETUP
# ==========================================
try:
    ssl._create_default_https_context = ssl._create_unverified_context
except AttributeError:
    pass

nltk.download('stopwords', quiet=True)
nltk.download('wordnet', quiet=True)

# ==========================================
# STEP 1: DOWNLOAD & FULL DATA LOADING
# ==========================================
url = "http://ai.stanford.edu/~amaas/data/sentiment/aclImdb_v1.tar.gz"
filename = "aclImdb_v1.tar.gz"

if not os.path.exists("aclImdb"):
    print("Downloading full dataset (50k reviews)...")
    urllib.request.urlretrieve(url, filename)
    with tarfile.open(filename, "r:gz") as tar:
        tar.extractall(filter='data')
    print("Download and extraction complete!")

def load_dataset(directory):
    data = []
    for label in ['pos', 'neg']:
        path = os.path.join("aclImdb", directory, label)
        for file in os.listdir(path):
            with open(os.path.join(path, file), "r", encoding="utf-8") as f:
                data.append({"review": f.read(), "label": 1 if label == 'pos' else 0})
    return pd.DataFrame(data)

print("Loading full training and testing sets...")
train_df = load_dataset("train")
test_df = load_dataset("test")

# ==========================================
# STEP 2: PREPROCESSING (Cleaning)
# ==========================================
stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

def clean_text(text):
    text = re.sub("[^a-zA-Z]", " ", text.lower())
    words = text.split()
    cleaned = [lemmatizer.lemmatize(w) for w in words if w not in stop_words]
    return " ".join(cleaned)

print("Cleaning 50,000 reviews (this will take a minute)...")
train_df['cleaned'] = train_df['review'].apply(clean_text)
test_df['cleaned'] = test_df['review'].apply(clean_text)

# ==========================================
# STEP 3: ENCODING (TF-IDF)
# ==========================================
print("Converting text to numbers...")
vectorizer = TfidfVectorizer(max_features=5000)
X_train = vectorizer.fit_transform(train_df['cleaned'])
X_test = vectorizer.transform(test_df['cleaned'])
y_train = train_df['label']
y_test = test_df['label']

# ==========================================
# STEP 4: TRAINING & EVALUATION
# ==========================================
print("Training Multinomial Naive Bayes model...")
model = MultinomialNB()
model.fit(X_train, y_train)

print("Evaluating model...")
predictions = model.predict(X_test)
accuracy = accuracy_score(y_test, predictions)

print("\n==========================================")
print(f"FINAL PROJECT ACCURACY: {accuracy * 100:.2f}%")
print("==========================================\n")
print("Detailed Report:")
print(classification_report(y_test, predictions, target_names=['Negative', 'Positive']))