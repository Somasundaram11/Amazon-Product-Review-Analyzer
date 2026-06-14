import streamlit as st
import pickle
import spacy

model = pickle.load(open("sentiment_model.pkl", "rb"))
tfidf = pickle.load(open("vectorizer.pkl", "rb"))

nlp = spacy.load("en_core_web_sm")

negations = {"not", "no", "never", "nor"}

def preprocess(text):

    doc = nlp(str(text).lower())

    tokens = []

    for token in doc:

        if token.is_punct:
            continue

        if token.text in negations:
            tokens.append(token.text)

        elif not token.is_stop:
            tokens.append(token.lemma_)

    return " ".join(tokens)

def predict_sentiment(review):

    clean_review = preprocess(review)

    vector = tfidf.transform([clean_review])

    prediction = model.predict(vector)

    probability = model.predict_proba(vector)

    confidence = max(probability[0]) * 100

    if prediction[0] == 1:
        sentiment = "Positive Review 😊"
    else:
        sentiment = "Negative Review 😞"

    return sentiment, confidence

st.set_page_config(
    page_title="Product Review Analyzer",
    page_icon="🛒"
)

st.title("Product Review Analyzer")

st.markdown(
    "Enter a customer review and predict whether it is Positive or Negative."
)

review = st.text_area(
    "Enter Product Review",
    height=150
)

if st.button("Predict Sentiment"):

    if review.strip() == "":
        st.warning("Please enter a review.")

    else:

        result, confidence = predict_sentiment(review)

        if "Positive" in result:
            st.success(result)

        else:
            st.error(result)

        st.info(f"Confidence Score: {confidence:.2f}%")

st.markdown("---")

st.caption("Built using NLP, spaCy, TF-IDF and Naive Bayes")
