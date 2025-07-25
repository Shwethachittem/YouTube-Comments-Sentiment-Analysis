import pandas as pd
import re
import nltk
import pickle
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score
from nltk.corpus import stopwords

# Download stopwords
nltk.download('stopwords')

# Load Dataset
df = pd.read_csv('youtube_comments_cleaned.csv')  # Change path if needed

# Normalize and clean the Sentiment column
df['Sentiment'] = df['Sentiment'].astype(str).str.strip().str.lower()
df = df[df['Sentiment'].isin(['positive', 'neutral', 'negative'])]

# Preprocessing function
def preprocess_text(text):
    text = str(text).lower()
    text = re.sub(r"http\S+|www\S+|https\S+", '', text)
    text = re.sub(r"[^a-zA-Z\s]", '', text)
    stop_words = set(stopwords.words('english'))
    words = [word for word in text.split() if word not in stop_words]
    return ' '.join(words)

# Clean comments
df['cleaned_comment'] = df['CommentText'].apply(preprocess_text)

# Encode sentiment
label_map = {'negative': 0, 'neutral': 1, 'positive': 2}
df['SentimentEncoded'] = df['Sentiment'].map(label_map)

# Feature and target
X = df['cleaned_comment']
y = df['SentimentEncoded']

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# TF-IDF vectorizer
tfidf = TfidfVectorizer(ngram_range=(1, 3), max_features=10000)
X_train_tfidf = tfidf.fit_transform(X_train)
X_test_tfidf = tfidf.transform(X_test)

# Logistic Regression model
lr_model = LogisticRegression(max_iter=200, multi_class='multinomial', solver='lbfgs')
lr_model.fit(X_train_tfidf, y_train)

# Evaluate
y_pred = lr_model.predict(X_test_tfidf)
print("Accuracy:", accuracy_score(y_test, y_pred) * 100)
print("\nClassification Report:\n", classification_report(y_test, y_pred, target_names=['Negative', 'Neutral', 'Positive']))

# ===  Pickle the model and vectorizer ===
with open("youtube_sentiment_model.pkl", "wb") as model_file:
    pickle.dump(lr_model, model_file)

with open("youtube_tfidf_vectorizer.pkl", "wb") as vec_file:
    pickle.dump(tfidf, vec_file)

print("\nModel and vectorizer saved successfully as .pkl files.")
