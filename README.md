---

# **Social Media Sentiment Analysis - README**

## **Project Objective**  
The project aims to analyze sentiment on Twitter, specifically focusing on detecting hate speech in tweets. Using natural language processing (NLP) techniques, the tweets are classified as either **racist** or **non-racist**, leveraging machine learning models trained on a labeled dataset.

---

## **Dataset Information**  
- **Source**: A dataset of 31,962 tweets labeled as racist or non-racist.  
- **Preprocessing Steps**:  
  - Removed Twitter handles, special characters, numbers, and punctuations.  
  - Tokenized tweets into individual words.  
  - Applied stemming to reduce words to their base form.  
- **Feature Extraction**: Utilized the Bag of Words (BoW) method for transforming text data into numerical form.  

---

## **Technologies Used**  
- **Programming Language**: Python  
- **Libraries**: Pandas, NumPy, Scikit-learn, NLTK, Matplotlib, Seaborn, WordCloud  

---

## **Steps Involved**  

### **1. Data Preprocessing**  
- **Text Cleaning**: Removed unnecessary elements like Twitter handles, special characters, and numbers.  
- **Tokenization**: Split sentences into individual words.  
- **Stemming**: Reduced words to their root forms for consistency.  

### **2. Exploratory Data Analysis (EDA)**  
- Generated word clouds and bar plots to visualize frequent words and hashtags.  
- Identified common trends in tweets with racist and non-racist sentiments.  

### **3. Feature Extraction**  
- Used the Bag of Words (BoW) method to convert text into a numerical representation.  

### **4. Model Training**  
- Split the dataset into training and testing sets.  
- Trained a logistic regression model on the extracted features.  

### **5. Model Evaluation**  
- Evaluated performance using accuracy and F1 score metrics.  

---

## **Key Functions**  

1. **`preprocess_text(text)`**:  
   - Cleans and preprocesses raw tweet text.  
   - Returns tokenized and stemmed words.  

2. **`extract_features(data)`**:  
   - Transforms preprocessed text into numerical form using the BoW model.  

3. **`train_model(X_train, y_train)`**:  
   - Trains a logistic regression model on the training data.  

4. **`evaluate_model(model, X_test, y_test)`**:  
   - Tests the model on unseen data and calculates evaluation metrics.  

---

## **Results & Observations**  
- The logistic regression model performed well, achieving:  
  - **Accuracy**: High classification accuracy on the test set.  
  - **F1 Score**: Demonstrated balanced precision and recall.  
- Visualization through word clouds highlighted dominant words in both racist and non-racist categories.  

---

## **Future Enhancements**  
1. **Advanced NLP Techniques**:  
   - Use transformer-based models like BERT for improved accuracy.  
   - Implement word embeddings (e.g., Word2Vec or GloVe) for richer feature extraction.  

2. **Enhanced Dataset**:  
   - Expand the dataset to include tweets with mixed sentiments.  
   - Include multilingual tweets for broader applicability.  

3. **Dashboard Integration**:  
   - Create an interactive dashboard to visualize sentiment trends in real-time.  

---

## **Dependencies**  
- Python 3.x  
- Pandas, NumPy  
- Scikit-learn  
- NLTK  
- Matplotlib, Seaborn  
- WordCloud  

---

## **How to Run**  

### **Setup**  
1. Install necessary libraries:  
   ```bash  
   pip install pandas numpy scikit-learn nltk matplotlib seaborn wordcloud  
   ```  
2. Place the dataset (`tweets.csv`) in the working directory.  

### **Run the Code**  
1. Execute the script for preprocessing, feature extraction, and model training:  
   ```bash  
   python sentiment_analysis.py  
   ```  
2. View results in the terminal or saved output files.  

---

## **Contributors**  
- **Aeni Parmar**  
  Data Analyst | [GitHub](https://github.com/AeniParmar)  

---
