# YouTube Comment Sentiment Analyzer

A web application that analyzes sentiments of comments on any YouTube video using **NLP** and **Machine Learning**. It classifies comments into **Positive 😊**, **Neutral 😐**, or **Negative 😞**.

---

## 1. Objective

To build an end-to-end application that:
- Accepts a YouTube video URL.
- Fetches comments using the YouTube Data API.
- Cleans and processes the comments.
- Classifies them into sentiment categories.
- Displays the results in a simple and interactive web interface.

---

## 2️. Model Training

### 🔹 Dataset
- Used a cleaned CSV dataset of YouTube comments labeled with sentiments (`Positive`, `Neutral`, `Negative`).

### 🔹 Preprocessing
Applied the following preprocessing steps:
- Lowercasing
- Removing URLs, punctuation, and special characters
- Removing stopwords (using NLTK)
- Stemming (using `PorterStemmer`)

### 🔹 Feature Extraction
- Used `TfidfVectorizer` with n-gram range of (1, 3) to convert text into numeric features.

### 🔹 Model
- Used `LogisticRegression` from `scikit-learn` with `multi_class='multinomial'`.

### 🔹 Output
- Saved the trained model and vectorizer using `joblib`:
  - `youtube_sentiment_model.pkl`
  - `youtube_tfidf_vectorizer.pkl`

---

## 3️. Web Application (Flask)

### 🔹 Inputs
- YouTube video URL

### 🔹 Backend Workflow
1. Extract video ID from URL.
2. Use YouTube Data API v3 to fetch **top-level comments**.
3. Preprocess the comments.
4. Convert to TF-IDF vectors.
5. Predict sentiments using the trained model.
6. Group results and display them by sentiment category.

### 🔹 Output
- Comments displayed under:  
  1. Positive  
  2. Neutral  
  3. Negative

---

