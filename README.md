# Sentiment Analysis of Movie Reviews
This project performs sentiment analysis on IMDb movie reviews using Logistic Regression, achieving **89.24% accuracy**. It includes a **Streamlit web app** for real-time sentiment predictions.

## Features
- Preprocesses text using NLTK stopwords and TF-IDF vectorization.
- Trains a Logistic Regression model with Scikit-learn.
- Visualizes performance with a confusion matrix.
- Deploys an interactive web interface via Streamlit in Google Colab.

## Dataset
- Uses the [IMDb Dataset](https://www.kaggle.com/datasets/lakshmi25npathi/imdb-dataset-of-50k-movie-reviews) (not included due to size/license; download from Kaggle).

## Requirements
```bash
pip install -r requirements.txt
