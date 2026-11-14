
# SMS Spam Detection Using Natural Language Processing

## Project Overview

This project focuses on building a machine learning-based SMS Spam Detection system using Natural Language Processing (NLP).

The objective is to classify SMS messages into two categories:
- **Ham:** Legitimate messages.
- **Spam:** Unwanted or promotional messages.

Different text representation techniques are explored to understand their impact on classification performance.

## Technologies Used

- Python
- Pandas and NumPy
- NLTK
- Scikit-learn
- Gensim
- Matplotlib and Seaborn (for visualizations)

## NLP Techniques Implemented

### 1. Bag of Words (BoW)
- Text representation using CountVectorizer.
- Feature extraction using word n-grams.
- Classification using Multinomial Naive Bayes.

### 2. TF-IDF
- Text representation using TF-IDF Vectorizer.
- Unigram and bigram feature extraction.
- Classification using Multinomial Naive Bayes.

### 3. Word2Vec
- Word embedding generation using Gensim.
- Training a Word2Vec model on the SMS corpus.
- Average word embeddings for document representation.
- Classification using Multinomial Naive Bayes.

## Project Workflow

1. Load the SMS Spam Collection dataset.
2. Perform text preprocessing.
3. Remove unwanted characters and stopwords.
4. Apply stemming.
5. Convert text into numerical features.
6. Train machine learning models.
7. Evaluate model performance.
8. Compare different NLP techniques.

## Dataset

Dataset: SMS Spam Collection

Source: UCI Machine Learning Repository

[Download the dataset](https://archive.ics.uci.edu/dataset/228/sms+spam+collection)

## Model Evaluation

The models will be evaluated using:

- Accuracy
- Precision
- Recall
- F1-score
- Confusion Matrix

| Technique | Algorithm | Accuracy |
|---|---|---|
| Bag of Words | Multinomial Naive Bayes | To be updated |
| TF-IDF | Multinomial Naive Bayes | To be updated |
| Word2Vec | Multinomial Naive Bayes | To be updated |

*Evaluation results will be updated after running the corrected notebook.*

## How to Run

1. Clone this repository.
2. Install the required dependencies.
3. Download the SMS Spam Collection dataset.
4. Open the notebook in Google Colab or Jupyter Notebook.
5. Execute the cells sequentially.

## Future Enhancements

- Experiment with Logistic Regression and SVM.
- Compare different Word2Vec configurations.
- Explore pretrained word embeddings.
- Build an interactive SMS Spam Detection application.

## Author

Vishal Peddaraju

GitHub: [vishalpeddaraju](https://github.com/vishalpeddaraju)

