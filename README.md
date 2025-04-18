# Sentiment_Analysis_

A comprehensive NLP project for sentiment and emotion classification using TF-IDF, lemmatization, and machine learning models including Logistic Regression, SVM, Random Forest, and Naive Bayes. The project also includes data preprocessing, visualization, and model evaluation.

### Project Purpose
---

This project aims to build a machine learning system capable of classifying text data based on sentiment (positive, negative, neutral) and emotion (joy, anger, sadness, etc.). By leveraging NLP and classification techniques, it supports various real-world applications such as customer feedback analysis, social media monitoring, recommendation systems, and mental health tracking.

### Key Techniques
---

**Data Preprocessing:**
- **Text Cleaning**: Removing punctuation, numbers, stopwords, and converting text to lowercase.
- **Lemmatization**: Applying WordNet lemmatizer to reduce words to their base forms.

**Feature Extraction:**
- **TF-IDF (Term Frequency-Inverse Document Frequency)**: Extracting features from text data, with TF-IDF weighing terms based on their importance in the dataset.
- **Count Vectorizer**: An alternative feature extraction method that counts the frequency of each term in the document.

**Model Training:**
- **Logistic Regression**: A linear model used for classification tasks.
- **Support Vector Machine (SVM)**: A supervised learning model that utilizes hyperplanes for data separation.
- **Random Forest**: An ensemble learning method that constructs multiple decision trees and combines their results.
- **Naive Bayes**: A probabilistic model based on Bayes' theorem, useful for text classification.

**Model Evaluation:**
- **Accuracy Score**: Measuring the proportion of correctly predicted labels.
- **Confusion Matrix**: A matrix used to evaluate the performance of classification models.
- **Classification Report**: A detailed report of precision, recall, and F1 score for each class.

**Hyperparameter Tuning:**
- **GridSearchCV**: Used to find the best hyperparameters for Logistic Regression, optimizing the model’s performance.

### Dataset Details
---

The dataset used in this project consists of text data labeled with sentiment and emotion categories. It includes two distinct datasets: one for emotion classification and another for sentiment classification.

#### **Emotion Dataset**:
- **Source**: [Kaggle - Sentiment and Emotion Analysis Dataset](https://www.kaggle.com/datasets/kushagra3204/sentiment-and-emotion-analysis-dataset)
- **Columns**:
  - `sentence`: The text of the sentence.
  - `emotion`: The emotion label for each sentence (e.g., happiness, sadness, anger, fear).
  
This dataset contains various sentences labeled with an emotion category, representing different moods or feelings expressed by people.

#### **Sentiment Dataset**:
- **Source**: [Kaggle - Sentiment and Emotion Analysis Dataset](https://www.kaggle.com/datasets/kushagra3204/sentiment-and-emotion-analysis-dataset)
- **Columns**:
  - `sentence`: The text of the sentence.
  - `sentiment`: The sentiment label for each sentence (e.g., positive, negative, neutral).
  
This dataset is focused on sentiment analysis, where sentences are labeled based on their overall sentiment, ranging from positive, neutral, or negative.

### Visualization
---

Throughout the project, data visualizations were used to understand the distribution of emotion and sentiment categories, as well as the relationship between text length and categories. Key plots include:
- Count plots of emotion and sentiment distributions.
- Box plots showing the distribution of text lengths by emotion and sentiment categories.

### Future Improvements
---

- Exploring advanced models like deep learning techniques (e.g., LSTM, BERT) for better performance.
- Experimenting with more sophisticated text preprocessing techniques such as stemming or handling domain-specific stopwords.
- Extending the models to support multi-lingual text classification.


---

**Tools Used:**
- **Python**: Core language for implementation.
- **NLP Libraries**: NLTK, scikit-learn, and pandas for text processing and model training.
- **Visualization**: Matplotlib and Seaborn for data visualization.
- **Machine Learning**: Logistic Regression, SVM, Random Forest, and Naive Bayes for classification.

---

Feel free to explore, contribute, and extend this project to various domains of natural language processing and text classification!
