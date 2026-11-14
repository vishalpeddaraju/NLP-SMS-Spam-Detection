# SMS Spam Detection Using Natural Language Processing (NLP)

## Project Overview

This project focuses on developing a Machine Learning-based SMS Spam Detection system using Natural Language Processing (NLP).

The objective is to classify SMS messages into two categories:

- **Ham:** Legitimate messages.
- **Spam:** Unwanted, fraudulent, or promotional messages.

The project explores different text representation techniques to understand how they influence the performance of machine learning models in text classification.

The following NLP techniques are implemented and compared:

1. Bag of Words (BoW)
2. Term Frequency–Inverse Document Frequency (TF-IDF)
3. Word2Vec

## Technologies Used

- **Programming Language:** Python
- **Data Processing:** Pandas, NumPy
- **Natural Language Processing:** NLTK
- **Machine Learning:** Scikit-learn
- **Word Embeddings:** Gensim
- **Data Visualization:** Matplotlib, Seaborn
- **Development Environment:** Google Colab / Jupyter Notebook

## NLP Techniques Implemented

### 1. Bag of Words (BoW)

Bag of Words converts text messages into numerical feature vectors based on word occurrence.

**Implementation:**
- Text representation using `CountVectorizer`.
- Feature extraction using bigrams.
- Maximum of 2,500 features.
- Binary feature representation.
- Classification using Multinomial Naive Bayes.

### 2. TF-IDF (Term Frequency–Inverse Document Frequency)

TF-IDF represents text based on the importance of words within a document relative to the complete collection of documents.

**Implementation:**
- Text representation using `TfidfVectorizer`.
- Unigram and bigram feature extraction.
- Maximum of 2,500 features.
- Classification using Multinomial Naive Bayes.

### 3. Word2Vec

Word2Vec is a word embedding technique that converts words into dense numerical vectors, capturing semantic and contextual relationships.

**Implementation:**
- Word embedding generation using Gensim.
- Training a Word2Vec model on the training SMS messages.
- Skip-gram architecture (`sg=1`).
- Vector size of 100 dimensions.
- Average word embeddings to generate document-level representations.
- Classification using Logistic Regression.

## Project Workflow

The project follows these major steps:

1. Load the SMS Spam Collection dataset.
2. Perform exploratory data analysis and examine class distribution.
3. Clean and preprocess SMS messages.
4. Convert text to lowercase.
5. Remove unwanted characters and stopwords.
6. Apply stemming using Porter Stemmer.
7. Split the dataset into training and testing sets.
8. Apply Bag of Words, TF-IDF, and Word2Vec techniques.
9. Train the respective machine learning classifiers.
10. Evaluate model performance using classification metrics.
11. Compare the results and visualize model performance.

## Dataset

**Dataset:** SMS Spam Collection

**Source:** UCI Machine Learning Repository

**Dataset Link:** https://archive.ics.uci.edu/dataset/228/sms+spam+collection

The dataset contains SMS messages labelled as either `ham` or `spam`.

The dataset is downloaded programmatically through the notebook, so manual dataset uploads are not required when using the provided Google Colab workflow.

## Model Evaluation

The models are evaluated using the following performance metrics:

- **Accuracy:** Measures the overall proportion of correctly classified messages.
- **Precision:** Measures how many messages predicted as spam are actually spam.
- **Recall:** Measures how many actual spam messages are correctly identified.
- **F1-Score:** Represents the harmonic mean of precision and recall.
- **Confusion Matrix:** Provides a detailed view of correct and incorrect predictions.

### Model Performance Comparison

The following table presents the performance of the three implemented NLP approaches on the test dataset.

| Model | Classification Algorithm | Accuracy | Precision | Recall | F1-Score |
|---|---|---:|---:|---:|---:|
| Bag of Words | Multinomial Naive Bayes | 96.68% | 99.12% | 75.84% | 85.93% |
| TF-IDF | Multinomial Naive Bayes | 97.22% | 100.00% | 79.19% | 88.39% |
| Word2Vec | Logistic Regression | 96.50% | 89.86% | 83.22% | 86.41% |

### Key Observations

- **TF-IDF achieved the highest accuracy of 97.22%** and the highest F1-score of 88.39%.
- **TF-IDF achieved 100% precision**, indicating that all messages predicted as spam in the test set were actually spam.
- **Word2Vec achieved the highest recall of 83.22%**, identifying a larger proportion of actual spam messages compared with the other two approaches.
- Bag of Words achieved an accuracy of 96.68%, demonstrating strong classification performance.

### Conclusion

Based on the experimental results, TF-IDF with Multinomial Naive Bayes delivered the best overall performance in terms of accuracy, precision, and F1-score.

However, Word2Vec with Logistic Regression achieved the highest recall, making it useful when identifying a greater proportion of spam messages is a priority.

The comparison demonstrates that different text representation techniques influence classification performance, and the most suitable approach depends on the evaluation metric and application requirements.

## Repository Structure

```text
NLP-SMS-Spam-Detection/
│
├── word2vec_fromscratch.ipynb
├── README.md
├── requirements.txt
├── .gitignore
│
└── screenshots/
    ├── preprocessing.png
    ├── bow-results.png
    ├── tfidf-results.png
    └── word2vec-results.png
```

*Note: The screenshots folder can be added to the repository to showcase model outputs and visualizations.*

## How to Run the Project

### Option 1: Google Colab (Recommended)

1. Clone or download this repository.
2. Open `word2vec_fromscratch.ipynb` in Google Colab.
3. Execute the notebook cells sequentially.
4. The SMS Spam Collection dataset will be downloaded automatically.
5. Run the preprocessing, feature extraction, model training, and evaluation sections.
6. Review the model comparison table and visualizations.

### Option 2: Local Environment

1. Clone the repository:

   ```bash
   git clone https://github.com/vishalpeddaraju/NLP-SMS-Spam-Detection.git
   ```

2. Navigate to the project directory:

   ```bash
   cd NLP-SMS-Spam-Detection
   ```

3. Install the required dependencies:

   ```bash
   pip install -r requirements.txt
   ```

4. Open the notebook using Jupyter Notebook:

   ```bash
   jupyter notebook
   ```

5. Execute the notebook cells sequentially.

*Note: The current dataset-loading code uses Google Colab's `/content` directory. For local execution, update the dataset storage path accordingly.*

## Future Enhancements

- Experiment with additional classification algorithms such as Support Vector Machines (SVM) and Random Forest.
- Explore different Word2Vec configurations and embedding dimensions.
- Experiment with pretrained word embeddings.
- Implement advanced NLP techniques such as GloVe and FastText.
- Build an interactive SMS Spam Detection application using Streamlit.
- Improve spam detection recall while maintaining high precision.

## Author

**Vishal Peddaraju**

GitHub: https://github.com/vishalpeddaraju

