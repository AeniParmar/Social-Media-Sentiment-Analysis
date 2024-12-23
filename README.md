_**Project Objective**_

The aim of this project is to detect hate speech in tweets. Specifically, a tweet is classified as containing hate speech if it exhibits a racist sentiment. The task involves building a model to classify tweets into two categories:

Label '1': The tweet contains racist sentiment.

Label '0': The tweet does not contain racist sentiment.

A labeled dataset of 31,962 tweets is provided for training the model. Each record in the dataset contains:
-Tweet ID
-Label (0 or 1)
-The tweet text

**Dataset Information** :The dataset is a CSV file containing the necessary information to train and test the hate speech detection model.

Data preprocessing is critical to clean and prepare the tweets for analysis and model building.

**Prerequisites**
_Python libraries_: pandas, numpy, matplotlib, seaborn, re, string, nltk, wordcloud, sklearn
Basic understanding of Natural Language Processing (NLP) and Machine Learning (ML).

**Steps Involved**
_1. Data Loading and Inspection_
Load the dataset using pandas.
Inspect the dataset structure with .info() and .head() methods.

_2. Data Cleaning_
Remove Twitter handles (e.g., @user) using regex.
Remove special characters, numbers, and punctuations.
Remove short words with fewer than three characters.
Tokenize and stem words using the PorterStemmer from the nltk library.
Reconstruct cleaned tweets into sentences.

_3. Data Visualization_
WordCloud Visualization
Generate word clouds for:
All cleaned tweets.
Non-racist tweets (label 0).
Racist tweets (label 1).

**Hashtag Analysis**
Extract hashtags from non-racist and racist tweets.
Identify the top 10 most frequent hashtags for each category and visualize them using bar plots.

_4. Feature Extraction_
Use CountVectorizer from sklearn to convert the text into a Bag of Words (BoW) representation.
Configure CountVectorizer with:
max_df=0.90
min_df=2
max_features=1000
stop_words='english'

_5. Train-Test Split_
Split the data into training and testing sets using train_test_split from sklearn:
Test size: 25%
Random state: 42

_6. Model Training and Testing_
Train a Logistic Regression model on the training set.
Evaluate the model using:
F1 Score
Accuracy Score

_7. Threshold Adjustment_
Use predicted probabilities to classify tweets.
Experiment with different thresholds (e.g., 0.3) to optimize the model's performance.

**Results**
The model performance is measured using F1 Score and Accuracy Score.
Fine-tuning of probability thresholds can improve classification results.

**Dependencies**
Python 3.x
pandas
numpy
matplotlib
seaborn
nltk
sklearn
wordcloud

**Observations**

Cleaning and preprocessing significantly impact the model's accuracy.
Most frequent hashtags provide insights into the nature of the tweets.
Logistic Regression performs reasonably well with BoW representation.
Adjusting probability thresholds can affect the balance between precision and recall.

**Notes**

The dataset may require further augmentation to improve model performance.

Consider exploring other feature extraction methods like TF-IDF or word embeddings.
