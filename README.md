# IMDb Movie Review Sentiment Analysis

A sentiment classifier trained on 50,000 IMDb movie reviews using TF-IDF and Naive Bayes.

## Overview
This project classifies movie reviews as positive or negative using the [Stanford IMDb dataset](https://ai.stanford.edu/~amaas/data/sentiment/). Reviews are cleaned with lemmatization and stop word removal, then vectorized with TF-IDF before being passed to a Multinomial Naive Bayes classifier.

The script automatically downloads the dataset on first run, so no manual setup is needed.

## Tech Stack

| | |
|---|---|
| Language | Python 3 |
| Libraries | scikit-learn, NLTK, pandas, matplotlib, seaborn |
| Text Preprocessing | Lowercasing, regex cleaning, stop word removal, lemmatization |
| Vectorization | TF-IDF (5000 features) |
| Model | Multinomial Naive Bayes |

## Getting Started

```bash
git clone https://github.com/siliconsagenerd/NLP_Movie_Reviews.git
cd NLP_Movie_Reviews
pip install -r requirements.txt
python main.py
```

> The dataset (~84MB) is downloaded automatically on the first run.

## Results

| | Negative | Positive |
|---|---|---|
| Precision | 83% | 85% |
| Recall | 86% | 82% |
| F1-Score | 84% | 84% |
| **Accuracy** | **84.00%** | |

## Confusion Matrix

![Confusion Matrix](images/confusion_matrix.png)

## Project Structure
```
NLP_Movie_Reviews/
├── main.py
├── README.md
├── requirements.txt
├── results.txt
├── data/
│   └── dataset_link.txt
└── images/
    └── confusion_matrix.png
```

## Dataset
The dataset is from the [Stanford AI Lab](https://ai.stanford.edu/~amaas/data/sentiment/) and contains 25,000 training and 25,000 test reviews.
