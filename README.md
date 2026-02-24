# NLP Sentiment Analysis: Movie Reviews
This project was developed for the **DLBAIPNLP01 – Project: NLP** course at IU International University of Applied Sciences.

## Project Overview
The goal of this system is to analyze movie reviews and determine whether the overall sentiment is positive or negative using Natural Language Processing (NLP).

## Technical Implementation
- **Data Collection:** Used the Stanford Large Movie Review Dataset (50,000 reviews).
- **Preprocessing:** Performed text cleaning, including lowercase conversion, removal of punctuation/stop words, and lemmatization.
- **Encoding:** Converted text data into numerical format using **TF-IDF Vectorization** (top 5000 features).
- **Model:** Trained a **Multinomial Naive Bayes** classifier using supervised learning.
- **Evaluation:** Progressively tested on larger datasets to assess generalization performance.

## Final Results
- **Overall Accuracy:** 84.00%
- Detailed performance metrics can be found in the `results.txt` file.