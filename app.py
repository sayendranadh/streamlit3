import string
import streamlit as st
import pandas as pd
import spacy
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, mean_squared_error, classification_report, roc_curve, auc, ConfusionMatrixDisplay, mean_absolute_error, r2_score
import matplotlib.pyplot as plt
import seaborn as sns
from imblearn.under_sampling import RandomUnderSampler
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from sklearn.feature_extraction.text import TfidfVectorizer
from wordcloud import WordCloud

# Load spaCy's English tokenizer
nlp = spacy.blank("en")

# Define text normalization function
def normalize(text_tokens):
    stop_words = spacy.lang.en.stop_words.STOP_WORDS
    punctuation = set(string.punctuation)
    return [word.lower() for word in text_tokens if word.lower() not in stop_words and word not in punctuation]

# Define a function to categorize sentiment
def categorize_sentiment(score):
    if score >= 0.05:
        return "Positive"
    elif score <= -0.05:
        return "Negative"
    else:
        return "Neutral"

# Streamlit app
st.title("🔍 Twitter Sentiment Analysis & ML Model Evaluation")

# Load pre-existing CSV file
csv_file_path = 'filtered_data.csv'
try:
    df = pd.read_csv(csv_file_path)

    st.markdown("## 📊 **Raw Data Preview**")
    st.dataframe(df.head())

    # Ensure 'text' column is string and handle missing values
    df['text'] = df['text'].astype(str).fillna('')

    # Tokenization & normalization
    df['tokenized_text'] = df['text'].apply(lambda x: [token.text for token in nlp(x) if token.text.strip()])
    df['normalized_text'] = df['tokenized_text'].apply(normalize)

    # **TF-IDF Vectorization**
    st.markdown("## 🔍 **TF-IDF Text Vectorization**")
    tfidf = TfidfVectorizer(max_features=500)  # Limit to top 500 words
    tfidf_matrix = tfidf.fit_transform(df['text'])

    # Convert TF-IDF matrix to DataFrame
    tfidf_df = pd.DataFrame(tfidf_matrix.toarray(), columns=tfidf.get_feature_names_out())

    # **Download TF-IDF Data**
    st.markdown("### 📥 **Download TF-IDF Features as CSV**")
    csv_data = tfidf_df.to_csv(index=False).encode('utf-8')
    st.download_button(label="⬇ Download CSV", data=csv_data, file_name="tfidf_features.csv", mime="text/csv")

    # **TF-IDF Feature Importance Visualization**
    st.markdown("## 🔥 **Most Important Words in TF-IDF**")
    tfidf_mean = tfidf_df.mean().sort_values(ascending=False).head(20)

    fig, ax = plt.subplots(figsize=(10, 6))
    sns.barplot(x=tfidf_mean.values, y=tfidf_mean.index, palette="magma", ax=ax)
    ax.set_title("Top 20 Most Important Words (TF-IDF) 🚀", fontsize=14)
    ax.set_xlabel("Average TF-IDF Score", fontsize=12)
    ax.set_ylabel("Words", fontsize=12)
    st.pyplot(fig)

    # **Word Cloud for TF-IDF**
    st.markdown("## 🌥 **TF-IDF Word Cloud**")
    wordcloud = WordCloud(width=800, height=400, background_color='black').generate_from_frequencies(tfidf_mean)

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.imshow(wordcloud, interpolation='bilinear')
    ax.axis("off")
    st.pyplot(fig)

    st.success("🎉 **TF-IDF Analysis & Visualizations Completed!** 🚀")

    # Word frequency visualization
    st.markdown("## 🔥 **Most Common Words in Tweets**")
    all_words = [word for tokens in df['normalized_text'] for word in tokens]
    word_freq = pd.Series(all_words).value_counts().head(20)

    fig, ax = plt.subplots(figsize=(10, 6))
    sns.barplot(x=word_freq.values, y=word_freq.index, palette="Blues_r", ax=ax)
    ax.set_title("Top 20 Most Common Words 🏆", fontsize=14)
    ax.set_xlabel("Frequency 📊", fontsize=12)
    ax.set_ylabel("Words 🔤", fontsize=12)
    st.pyplot(fig)

    # Sentiment Analysis using VADER
    analyzer = SentimentIntensityAnalyzer()
    df['sentiment_score'] = df['text'].apply(lambda x: analyzer.polarity_scores(x)['compound'])
    df['sentiment_category'] = df['sentiment_score'].apply(categorize_sentiment)

    # Sentiment Distribution
    st.markdown("## 📈 **Tweet Sentiment Analysis**")
    sentiment_counts = df['sentiment_category'].value_counts()

    fig, ax = plt.subplots(figsize=(8, 6))
    sns.barplot(x=sentiment_counts.index, y=sentiment_counts.values, palette="viridis", ax=ax)
    ax.set_title("Tweet Sentiment Categories 🎭", fontsize=14)
    st.pyplot(fig)

    # 📌 Ensure sentiment_category is created correctly (AFTER sentiment analysis)
    if 'sentiment_category' not in df.columns:
        st.error("⚠️ Sentiment analysis not performed yet! Please check sentiment score calculations.")

    # 📌 Ensure normalized_text is a list before exploding (AFTER text normalization)
    df['normalized_text'] = df['normalized_text'].apply(lambda x: x if isinstance(x, list) else [])

    # 🔥 Word Cloud Based on Sentiment
    st.markdown("## 🌥 Word Cloud Based on Sentiment")
    sentiment_choice = st.selectbox("Select Sentiment for Word Cloud", ["All", "Positive", "Negative", "Neutral"])

    # Debugging: Print unique sentiment categories
    st.write("Unique Sentiment Categories:", df['sentiment_category'].unique())

    if sentiment_choice != "All":
        words_filtered = df[df['sentiment_category'] == sentiment_choice]['normalized_text'].explode()
    else:
        words_filtered = df['normalized_text'].explode()

    # Handle potential issues with empty data
    if not words_filtered.empty:
        word_freq_filtered = pd.Series(words_filtered).value_counts()
        wordcloud_filtered = WordCloud(width=800, height=400, background_color='black').generate_from_frequencies(word_freq_filtered)

        fig, ax = plt.subplots(figsize=(10, 5))
        ax.imshow(wordcloud_filtered, interpolation="bilinear")
        ax.axis("off")
        st.pyplot(fig)
    else:
        st.warning(f"⚠️ No words found for {sentiment_choice} sentiment.")


    # Convert timestamp to datetime
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    sentiment_over_time = df.groupby(df['timestamp'].dt.date)['sentiment_score'].mean()

    st.markdown("## 📅 **Sentiment Trend Over Time**")
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(sentiment_over_time.index, sentiment_over_time.values, marker='o', color='blue')
    ax.set_title("Average Sentiment Over Time 📉📈", fontsize=14)
    ax.set_xlabel("Date 📅", fontsize=12)
    ax.set_ylabel("Sentiment Score ❤️💔", fontsize=12)
    ax.grid(True)
    st.pyplot(fig)

    # Correlation Heatmap
    st.markdown("## 🧩 **Feature Correlation Heatmap**")
    correlation_matrix = df[['likes', 'replies', 'retweets', 'quotes', 'sentiment_score']].corr()

    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(correlation_matrix, annot=True, cmap="coolwarm", fmt=".2f", ax=ax)
    ax.set_title("Feature Correlation 🔥", fontsize=14)
    st.pyplot(fig)

    # Feature Engineering
    df['isReply'] = df['isReply'].astype(int)
    df['isRetweet'] = df['isRetweet'].astype(int)
    df['isQuote'] = df['isQuote'].astype(int)
    df['verified'] = df['verified'].astype(int)

    st.success("🎉 **Data Processing Completed Successfully!** Ready for Machine Learning 🔥")

    features = df[['replies', 'retweets', 'quotes', 'isReply', 'isRetweet', 'isQuote', 'verified', 'sentiment_score']]
    target = df['likes']

    # Fix class imbalance
    binary_target = (target > target.mean()).astype(int)
    rus = RandomUnderSampler(random_state=42)
    X_resampled, y_resampled = rus.fit_resample(features, binary_target)

    # Model Selection
    model_option = st.selectbox("📌 Choose an ML Model", ["Linear Regression", "Logistic Regression", "Random Forest", "XGBoost"])

    # Train-Test Split
    X_train, X_test, y_train, y_test = train_test_split(X_resampled, y_resampled, test_size=0.2, random_state=42)

    # **Linear Regression for Predicting Likes**
    if model_option == "Linear Regression":
        # **Regression Model**
        model = LinearRegression()
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        # **Metrics**
        mse = mean_squared_error(y_test, y_pred)
        mae = mean_absolute_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)

        # **Display Regression Results**
        st.markdown(f"### 📊 {model_option} Results")
        st.write(f"✅ **Mean Squared Error (MSE):** {mse:.4f}")
        st.write(f"✅ **Mean Absolute Error (MAE):** {mae:.4f}")
        st.write(f"✅ **R² Score:** {r2:.4f}")

        # **Actual vs Predicted Scatter Plot**
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.scatter(y_test, y_pred, alpha=0.6, color="blue", label="Predicted")
        ax.plot(y_test, y_test, color="red", linestyle="--", label="Perfect Fit")
        ax.set_xlabel("Actual Likes")
        ax.set_ylabel("Predicted Likes")
        ax.set_title("Actual vs Predicted Likes")
        ax.legend()
        st.pyplot(fig)

    else:
        # **Classification Model Selection**
        if model_option == "Logistic Regression":
            model = LogisticRegression(max_iter=500)
        elif model_option == "Random Forest":
            model = RandomForestClassifier(n_estimators=100, random_state=42)
        elif model_option == "XGBoost":
            model = XGBClassifier(n_estimators=100, learning_rate=0.1, max_depth=5, random_state=42)

        # **Train & Predict**
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)

        # **Display Classification Results**
        st.markdown(f"### 📊 {model_option} Results")
        st.write(f"✅ **Accuracy:** {accuracy:.4f}")

        # **Classification Report**
        st.write("### 📑 Classification Report")
        st.text(classification_report(y_test, y_pred))

        # **Confusion Matrix**
        st.write("### 📌 Confusion Matrix")
        fig, ax = plt.subplots(figsize=(6, 6))
        ConfusionMatrixDisplay.from_estimator(model, X_test, y_test, ax=ax, cmap="Blues")
        plt.title(f"Confusion Matrix - {model_option}")
        st.pyplot(fig)

        # **ROC Curve (only for binary classification)**
        if hasattr(model, "predict_proba"):
            y_probs = model.predict_proba(X_test)[:, 1]  # Probability of positive class
        else:
            y_probs = model.decision_function(X_test)  # For models without probability output

        fpr, tpr, _ = roc_curve(y_test, y_probs)
        roc_auc = auc(fpr, tpr)

        st.write("### 📈 ROC Curve")
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.plot(fpr, tpr, color="blue", lw=2, label=f"AUC = {roc_auc:.2f}")
        ax.plot([0, 1], [0, 1], color="gray", linestyle="--")
        ax.set_xlabel("False Positive Rate")
        ax.set_ylabel("True Positive Rate")
        ax.set_title(f"ROC Curve - {model_option}")
        ax.legend()
        st.pyplot(fig)

        # **Feature Importance (only for Random Forest & XGBoost)**
        if model_option in ["Random Forest", "XGBoost"]:
            st.write("### 🔥 Feature Importance")
            importance = model.feature_importances_
            feature_names = features.columns

            fig, ax = plt.subplots(figsize=(10, 6))
            sns.barplot(x=importance, y=feature_names, ax=ax, palette="coolwarm")
            ax.set_title("Feature Importances")
            ax.set_xlabel("Importance Score")
            ax.set_ylabel("Features")
            plt.grid(True)
            st.pyplot(fig)

    # Download Processed Data
    processed_csv = 'processed_tweets.csv'
    df.to_csv(processed_csv, index=False)

    st.download_button(
        label="📥 Download Processed Data",
        data=open(processed_csv, 'rb').read(),
        file_name='processed_tweets.csv',
        mime='text/csv'
    )

except FileNotFoundError:
    st.error("⚠️ File not found: Please check if 'filtered_data.csv' exists.")
