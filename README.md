# 🛒 Amazon Product Review Analyzer

## Overview
Amazon Product Review Analyzer is a Natural Language Processing (NLP) application that analyzes customer reviews and predicts whether a review is **Positive** or **Negative**. The system uses text preprocessing techniques, TF-IDF feature extraction, and a Naive Bayes machine learning model to perform sentiment classification.

The application is deployed using Streamlit, allowing users to enter product reviews and receive instant sentiment predictions along with a confidence score.

## Features
* Real-time sentiment prediction
* Positive and Negative review classification
* Confidence score for predictions
* Text preprocessing using spaCy
* Interactive Streamlit web interface
* Trained machine learning model using Naive Bayes

## Project Workflow
User Review
     ↓
Text Preprocessing
     ↓
Tokenization
     ↓
Stopword Removal
     ↓
Lemmatization
     ↓
TF-IDF Vectorization
     ↓
Naive Bayes Classifier
     ↓
Sentiment Prediction
     ↓
Display Result

## Technologies Used

* Python
* Pandas
* NumPy
* spaCy
* Scikit-learn
* Streamlit
* TF-IDF Vectorization
* Naive Bayes Classifier

## Dataset

The project uses the Amazon Reviews Dataset containing customer product reviews labeled as positive or negative.

Dataset files:

* train.ft.txt
* test.ft.txt


## Machine Learning Pipeline

### Data Preprocessing

* Lowercase Conversion
* Punctuation Removal
* Stopword Removal
* Negation Handling
* Lemmatization

### Feature Extraction

TF-IDF (Term Frequency–Inverse Document Frequency) is used to convert textual reviews into numerical feature vectors suitable for machine learning algorithms.

### Classification

The Multinomial Naive Bayes algorithm is trained on TF-IDF features to classify reviews as Positive or Negative.

## Installation

Clone the repository:

git clone <repository-url>
cd Amazon-Product-Review-Analyzer

Install required dependencies:

pip install -r requirements.txt


Download the spaCy language model:

python -m spacy download en_core_web_sm

## Running the Application

Start the Streamlit application:

streamlit run app.py

The application will open in your browser at:

http://localhost:8501

## Example Predictions

### Positive Review

Input:
Amazing quality and fast delivery.

Output:

Positive Review 😊
Confidence Score: 96%

### Negative Review

Input:
Worst product ever. Waste of money.

Output:
Negative Review 😞
Confidence Score: 94%

## Project Structure
Amazon-Product-Review-Analyzer
│
├── app.py
├── sentiment_model.pkl
├── vectorizer.pkl
├── requirements.txt
├── README.md
├── train.ft.txt
├── test.ft.txt
└── notebook.ipynb

## Future Enhancements

* Multi-class sentiment classification
* Aspect-based sentiment analysis
* Review summarization
* Product recommendation integration
* Advanced transformer-based models (BERT)

## Author

**M. Somasundaram**

B.Sc. Computer Science (AI & Data Science)

GitHub: https://github.com/Somasundaram11

LinkedIn: [www.linkedin.com/in/somasundaram-m-49535032a](http://www.linkedin.com/in/somasundaram-m-49535032a)

Email: [somasundaramm18@gmail.com](mailto:somasundaramm18@gmail.com)

## License

This project is developed for educational and learning purposes.
