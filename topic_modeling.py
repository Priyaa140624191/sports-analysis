import re
import nltk
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation

# Download stopwords if not already downloaded
nltk.download('stopwords')
stop_words = set(stopwords.words('english'))


# Preprocessing function
def preprocess_text(text):
    text = re.sub(r"http\S+", "", text)  # Remove URLs
    text = re.sub(r"[^a-zA-Z\s]", "", text)  # Remove special characters and numbers
    text = text.lower().strip()  # Lowercase and remove extra spaces
    text = " ".join([word for word in text.split() if word not in stop_words])  # Remove stopwords
    return text


# Function to run LDA topic modeling
def lda_topic_modeling(text_data, n_topics=5, no_top_words=10):
    # Preprocess the text data
    cleaned_texts = [preprocess_text(text) for text in text_data]

    # Vectorize the text using CountVectorizer
    vectorizer = CountVectorizer(max_df=0.9, min_df=2, stop_words='english')
    dtm = vectorizer.fit_transform(cleaned_texts)

    # LDA model
    lda = LatentDirichletAllocation(n_components=n_topics, random_state=42)
    lda.fit(dtm)

    # Display topics
    topics = {}
    for topic_idx, topic in enumerate(lda.components_):
        topics[f"Topic {topic_idx}"] = " ".join(
            [vectorizer.get_feature_names_out()[i] for i in topic.argsort()[:-no_top_words - 1:-1]])

    return topics