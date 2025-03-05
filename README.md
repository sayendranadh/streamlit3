---

# 🔍 Twitter Sentiment Analysis & ML Model Evaluation 🚀

![Twitter Sentiment Analysis](https://img.shields.io/badge/Twitter-Sentiment%20Analysis-blue?style=for-the-badge&logo=twitter)  
📌 **A powerful Streamlit-based web application for analyzing Twitter data, performing sentiment analysis, extracting key insights using TF-IDF, and building machine learning models to predict engagement metrics (likes, retweets, replies, etc.).**

---

## 📜 **Table of Contents**  
- [📌 Features](#-features)  
- [🚀 Installation](#-installation)  
- [📂 Dataset Requirements](#-dataset-requirements)  
- [⚙️ How It Works](#️-how-it-works)  
- [📊 Data Processing & Visualization](#-data-processing--visualization)  
- [🤖 Machine Learning Models](#-machine-learning-models)  
- [📥 Downloading Processed Data](#-downloading-processed-data)  
- [📸 Screenshots](#-screenshots)  
- [🛠 Technologies Used](#-technologies-used)  
- [🙌 Contributing](#-contributing)  
- [📜 License](#-license)  

---

## 📌 **Features**  
✅ **Load Twitter Data from CSV** 📂  
✅ **Text Preprocessing & Normalization** (Tokenization, Stopword Removal, etc.)  
✅ **TF-IDF Vectorization for Keyword Extraction** 🔠  
✅ **WordCloud & Bar Charts for Word Frequency Analysis** 🌥📊  
✅ **VADER Sentiment Analysis (Positive, Negative, Neutral)** 🎭  
✅ **Sentiment Trend Over Time** 📉  
✅ **Correlation Heatmap of Twitter Features** 🔥  
✅ **Machine Learning Models for Engagement Prediction** 🧠  
✅ **Confusion Matrix, ROC Curve, and Feature Importance Analysis** 📈  
✅ **Download Processed Data & TF-IDF Features as CSV** 📥  

---

## 🚀 **Installation**  

### 📌 **1️⃣ Clone the Repository**  
```bash
git clone https://github.com/yourusername/twitter-sentiment-ml.git
cd twitter-sentiment-ml
```

### 📌 **2️⃣ Create a Virtual Environment**  
```bash
python -m venv venv
source venv/bin/activate   # MacOS/Linux
venv\Scripts\activate      # Windows
```

### 📌 **3️⃣ Install Dependencies**  
```bash
pip install -r requirements.txt
```

### 📌 **4️⃣ Run the Streamlit App**  
```bash
streamlit run app.py
```
📌 Open the URL in your browser: **https://app3-xv4r4cqlzqddwfyrux9qgv.streamlit.app/**  

---

## 📂 **Dataset Requirements**  
The application requires a CSV file named **`filtered_data.csv`** with the following columns:

| Column Name        | Description |
|--------------------|-------------|
| `text`            | Tweet content |
| `timestamp`       | Date & time of the tweet |
| `likes`           | Number of likes on the tweet |
| `retweets`        | Number of retweets |
| `replies`         | Number of replies |
| `quotes`          | Number of quote tweets |
| `isReply`         | 1 if it's a reply, 0 otherwise |
| `isRetweet`       | 1 if it's a retweet, 0 otherwise |
| `isQuote`         | 1 if it's a quote tweet, 0 otherwise |
| `verified`        | 1 if the user is verified, 0 otherwise |

---

## ⚙️ **How It Works**  

1️⃣ **Load Data** 📂  
   - Reads the Twitter dataset (`filtered_data.csv`) into a Pandas DataFrame.  
   - Prepares text data for analysis.  

2️⃣ **Preprocessing & Tokenization** 📝  
   - Tokenizes text using `spaCy`.  
   - Removes stopwords and punctuation.  

3️⃣ **TF-IDF Vectorization** 🔠  
   - Converts tweets into numerical features using **TF-IDF (Term Frequency - Inverse Document Frequency)**.  
   - Extracts the **top 500 important words** from the dataset.  

4️⃣ **Sentiment Analysis** 🎭  
   - Uses **VADER Sentiment Analysis** to categorize tweets into **Positive, Negative, or Neutral**.  
   - Plots sentiment distribution using Seaborn.  

5️⃣ **Machine Learning Model Training** 🧠  
   - Trains **Linear Regression, Logistic Regression, Random Forest, and XGBoost** models to predict engagement metrics.  
   - Evaluates models with **MSE, Accuracy, Confusion Matrix, and ROC Curve**.  

6️⃣ **Download Processed Data** 📥  
   - Allows users to download cleaned & processed data for further analysis.  

---

## 📊 **Data Processing & Visualization**  

### 🔍 **TF-IDF Feature Importance**  
- Displays the **Top 20 most important words** in the dataset using a bar chart.  

### 🌥 **WordCloud for Top Words**  
- Generates a **WordCloud** to visually represent the most frequent words in tweets.  

### 📉 **Sentiment Trend Over Time**  
- Plots **Average Sentiment Score** over time to identify trends in public opinion.  

### 🔥 **Feature Correlation Heatmap**  
- Shows how different features (likes, retweets, replies) correlate with each other.  

---

## 🤖 **Machine Learning Models**  

| Model                  | Type           | Usage |
|------------------------|---------------|---------------------------------------------------|
| **Linear Regression**  | Regression    | Predicts **number of likes** on a tweet. |
| **Logistic Regression** | Classification | Predicts if a tweet will get **above-average likes**. |
| **Random Forest**      | Classification | Predicts high-engagement tweets using ensemble learning. |
| **XGBoost**           | Classification | Predicts engagement with boosting techniques. |

---

## 📥 **Downloading Processed Data**  
The app allows you to download:  
✅ **TF-IDF Features** in CSV format.  
✅ **Processed Twitter Data** with sentiment scores.  

---

## 🛠 **Technologies Used**  
✅ **Python** 🐍  
✅ **Streamlit** 🎨  
✅ **Pandas, NumPy** 🏗  
✅ **spaCy** 🔠 (Text Processing)  
✅ **VADER Sentiment Analysis** 🎭  
✅ **Matplotlib, Seaborn** 📊 (Data Visualization)  
✅ **Scikit-learn, XGBoost** 🤖 (Machine Learning)  
✅ **WordCloud** 🌥 (Text Visualization)  

---

## 🙌 **Contributing**  
📌 Want to improve this project? Contributions are welcome!  
- Fork this repository.  
- Create a feature branch (`git checkout -b new-feature`).  
- Commit your changes (`git commit -m "Added new feature"`).  
- Push to the branch (`git push origin new-feature`).  
- Submit a pull request! 🚀  

---

## 📜 **License**  
This project is licensed under the **MIT License**.  

---

🚀 **Enjoy Analyzing Twitter Data with AI & Machine Learning!** 🎉  

---
