# Sentiment Analysis on Spotify App Reviews

## Overview

This project focuses on sentiment analysis of user reviews for the **Spotify mobile application**, obtained from Google Play Store. The analysis involves web scraping, data preprocessing, and classification using machine learning techniques.

## Features
* Web Scraping: Extracts user reviews from Google Play Store.

* Preprocessing: Cleans and prepares textual data for sentiment analysis.

* Sentiment Classification: Categorizes reviews into positive, neutral, or negative using machine learning models.

* Model Evaluation: Assesses the performance of the trained model using accuracy and other metrics.

## Technologies Used
* Python

* BeautifulSoup, Selenium (for web scraping)

* Pandas, NumPy (for data manipulation)

* NLTK, TextBlob (for text processing)

* Scikit-learn, TensorFlow (for model training and evaluation)

** Installation
* Clone this repository :

 ``` https://github.com/mfatarsyah/Proyek-Analisis-Sentimen-Dicoding.git ```
* Install dependencies:

 ``` pip install -r requirements.txt```

## Model Training

* Model 1: SVM with TF-IDF

The Support Vector Machine (SVM) model was applied using a TF-IDF-based feature representation to process text.
SMOTE was applied to the training data to address class imbalance issues.

* Model 2: Random Forest with TF-IDF
  
The Random Forest model was used with a TF-IDF-based feature representation.

* Model 3: Deep Learning with LSTM
  
A neural network model based on LSTM was used to process text.
The model was trained with data processed using Tokenizer and padding.

## Results

After training all three models, the evaluation results showed that the best model was selected based on the highest accuracy. A comparison between the actual labels and predictions for the best-performing model is presented to demonstrate the model's performance in predicting sentiment in reviews.
