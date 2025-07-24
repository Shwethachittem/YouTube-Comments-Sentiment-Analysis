from flask import Flask, render_template, request
from googleapiclient.discovery import build
import pickle, re
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
import nltk

nltk.download('stopwords')

# Load sentiment model and vectorizer
with open("youtube_sentiment_model.pkl", "rb") as f:
    model = pickle.load(f)
with open("youtube_tfidf_vectorizer.pkl", "rb") as f:
    vectorizer = pickle.load(f)

stop_words = set(stopwords.words('english'))
stemmer = PorterStemmer()

# === Text preprocessing ===
def clean_text(text):
    text = text.lower()
    text = re.sub(r"http\S+|www\S+|@\w+|#\w+", '', text)
    text = re.sub(r"[^\w\s]", '', text)
    text = re.sub(r"\d+", "", text)
    tokens = [stemmer.stem(word) for word in text.split() if word not in stop_words]
    return ' '.join(tokens)

# === Sentiment prediction ===
def predict_sentiment(comment):
    cleaned = clean_text(comment)
    vectorized = vectorizer.transform([cleaned])
    prediction = model.predict(vectorized)[0]

    if prediction == 2:
        return "Positive 😊"
    elif prediction == 0:
        return "Negative 😞"
    else:
        return "Neutral 😐"

# === YouTube comment extractor ===
def get_video_comments(video_id):
    api_key = "AIzaSyBA3b5meeHY8esIGjGoBt64cSp8f9GDfXk"  # 🔒 Replace this with your actual API key
    youtube = build('youtube', 'v3', developerKey=api_key)
    comments = []
    next_page_token = None

    while True:
        request = youtube.commentThreads().list(
            part='snippet',
            videoId=video_id,
            maxResults=100,
            textFormat='plainText',
            pageToken=next_page_token
        )
        response = request.execute()

        for item in response.get('items', []):
            comment = item['snippet']['topLevelComment']['snippet']['textDisplay']
            comments.append(comment)

        next_page_token = response.get('nextPageToken')
        if not next_page_token:
            break

    return comments

# === Video ID extractor ===
def extract_video_id(url):
    if "watch?v=" in url:
        return url.split("watch?v=")[-1].split("&")[0]
    elif "youtu.be/" in url:
        return url.split("youtu.be/")[-1].split("?")[0]
    return url

# === Flask app ===
app = Flask(__name__)

@app.route("/", methods=["GET", "POST"])
def index():
    positive_comments = []
    negative_comments = []
    neutral_comments = []

    if request.method == "POST":
        url = request.form["video_url"]
        video_id = extract_video_id(url)
        comments = get_video_comments(video_id)

        for comment in comments:
            sentiment = predict_sentiment(comment)
            if sentiment == "Positive 😊":
                positive_comments.append(comment)
            elif sentiment == "Negative 😞":
                negative_comments.append(comment)
            else:
                neutral_comments.append(comment)

    return render_template("index.html",
                           positives=positive_comments,
                           negatives=negative_comments,
                           neutrals=neutral_comments)

if __name__ == "__main__":
    app.run(debug=True)
