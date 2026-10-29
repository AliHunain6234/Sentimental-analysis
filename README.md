# Sentiment Analysis of Musical Instrument Reviews

A machine-learning project (5th semester, AI course) that classifies Amazon musical-instrument reviews as **negative, neutral or positive** using text preprocessing, TF-IDF features and scikit-learn classifiers.

## Files

| File | What it is |
|---|---|
| `Final Project.ipynb` | The full workflow: cleaning, exploration, features, models, evaluation |
| `Musical_instruments_reviews.csv` | The dataset: 10,261 reviews (columns: `reviewerID`, `asin`, `reviewerName`, `helpful`, `reviewText`, `overall`, `summary`, `unixReviewTime`, `reviewTime`) |
| `AI PROJECT.docx` | Short project description |

## What the notebook does

1. **Loads and cleans the data.** Missing review text is filled with an empty string, and `reviewText` and `summary` are joined into one `reviews` column.
2. **Labels the sentiment from the star rating.** A rating above 3 is positive, a rating of 3 is neutral, and below 3 is negative. The classes are very uneven: 9,022 positive, 772 neutral, 467 negative.
3. **Cleans the text.** Lower-casing, removing punctuation and other noise, and removing English stop-words (the word "not" is kept because it flips meaning), using NLTK.
4. **Explores the data.** Rating distribution, sentiment counts, review length and word counts, TextBlob polarity scores and word clouds.
5. **Builds features.** TF-IDF with bigrams, limited to 5,000 features.
6. **Balances the classes** with SMOTE so each class has 9,022 samples, then splits 75% train / 25% test.
7. **Compares models** on the test split:

| Model | Test accuracy |
|---|---|
| Decision Tree | 0.821 |
| K-Nearest Neighbors | 0.851 |
| Logistic Regression | 0.882 |
| Logistic Regression after a grid search (`C`, penalty) | **0.953** |

8. **Evaluates the best model** with a confusion matrix and a classification report (precision, recall and F1 are about 0.93 to 0.98 for the classes).

## Run it

```bash
pip install pandas numpy scikit-learn nltk textblob wordcloud matplotlib imbalanced-learn jupyter
jupyter notebook "Final Project.ipynb"
```

The first run downloads the NLTK data (punkt, stopwords) it needs. Keep the CSV in the same folder as the notebook.

## Things to keep in mind

- **The 95.3% figure is optimistic.** SMOTE is applied to the whole dataset *before* the train/test split, so synthetic copies of the same review can land on both sides. Doing the split first and oversampling only the training part would give a more honest number.
- Neutral and negative reviews are rare in the raw data, so accuracy on balanced data hides how the model does on the real, mostly positive distribution.
- The grid search printed many failed fits because some penalty and solver combinations are invalid; the best setting it found was used.
- The notebook outputs were produced on the author's computer, so some file paths in the saved outputs point to it.
