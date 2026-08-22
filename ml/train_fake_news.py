import pandas as pd
import numpy as np
import re
import string
import pickle
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, confusion_matrix, roc_curve

# ==========================================
# 1. NLTK Downloads & Initialization
# ==========================================
# We must download the English dictionaries required for text cleaning
nltk.download('stopwords')
nltk.download('wordnet')

# Initialize the lemmatizer
lemmatizer = WordNetLemmatizer()
# Load English stop words (e.g., 'the', 'is', 'in')
stop_words = set(stopwords.words('english'))

# ==========================================
# 2. Text Preprocessing Function
# ==========================================
def clean_text(text):
    """
    Cleans raw text data to improve AI training accuracy.
    """
    # 1. Lowercase: Make all text lowercase so "Apple" and "apple" are treated the same
    text = text.lower()
    
    # 2. Remove URLs, HTML tags, and weird brackets using Regular Expressions (Regex)
    # ADDED 'r' BEFORE STRINGS HERE
    text = re.sub(r'\[.*?\]', '', text)
    text = re.sub(r'https?://\S+|www\.\S+', '', text)
    text = re.sub(r'<.*?>+', '', text)
    
    # 3. Remove Punctuation (commas, periods, exclamation marks)
    text = re.sub(r'[%s]' % re.escape(string.punctuation), '', text)
    
    # 4. Remove newline characters and numbers
    text = re.sub(r'\n', ' ', text)
    text = re.sub(r'\w*\d\w*', '', text)
    
    # 5. Tokenization & Stopword Removal & Lemmatization
    # Split text into single words (tokens)
    words = text.split()
    # Keep the word only if it's not a useless stopword, and convert it to its root form (e.g., "running" -> "run")
    cleaned_words = [lemmatizer.lemmatize(word) for word in words if word not in stop_words]
    
    # Join the cleaned words back into a single string
    return ' '.join(cleaned_words)

# ==========================================
# 3. Loading and Preparing the Dataset
# ==========================================
def load_and_prepare_data():
    print("[INFO] Loading datasets from CSV files...")
    # Load the CSV files into Pandas DataFrames
    # ADDED 'r' BEFORE FILE PATHS HERE
    fake_df = pd.read_csv(r'C:\Users\admin\Desktop\Fake-News-Phishing-Detector\datasets\Fake.csv')
    true_df = pd.read_csv(r'C:\Users\admin\Desktop\Fake-News-Phishing-Detector\datasets\True.csv')
    
    try:
        local_true_df = pd.read_csv(r'C:\Users\admin\Desktop\Fake-News-Phishing-Detector\datasets\local_real_news.csv')
        true_df = pd.concat([true_df, local_true_df], axis=0)
    except FileNotFoundError:
        pass
    
    # Add a target label column. 0 = Fake, 1 = Real
    fake_df['label'] = 0
    true_df['label'] = 1
    
    # Combine the two datasets into one large dataset
    df = pd.concat([fake_df, true_df], axis=0)
    
    # Shuffle the dataset so Real and Fake are mixed randomly
    df = df.sample(frac=1).reset_index(drop=True)
    
    # Combine Title and Text into a single column for maximum context
    df['content'] = df['title'] + " " + df['text']
    
    # Keep only the columns we need to save memory
    df = df[['content', 'label']]
    
    print("[INFO] Cleaning text data (This may take a few minutes)...")
    # Apply our clean_text function to every row in the dataset
    df['content'] = df['content'].apply(clean_text)
    
    return df

# ==========================================
# 4. Training the Machine Learning Model
# ==========================================
def train_model():
    # 1. Get the cleaned data
    df = load_and_prepare_data()
    
    # Separate the features (X) and the target labels (Y)
    X = df['content']
    Y = df['label']
    
    # 2. Train / Test Split
    # Split data: 80% for training the AI, 20% for testing it on unseen data
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)
    
    print("[INFO] Vectorizing text using TF-IDF...")
    # 3. TF-IDF Vectorization
    # max_features=5000 means we only care about the top 5000 most important words to save memory
    vectorizer = TfidfVectorizer(max_features=5000)
    
    # Learn the vocabulary from the training set and transform the text into numbers
    X_train_vectorized = vectorizer.fit_transform(X_train)
    # Transform the test set using the ALREADY LEARNED vocabulary
    X_test_vectorized = vectorizer.transform(X_test)
    
    print("[INFO] Training Logistic Regression Model...")
    # 4. Initialize and train the Logistic Regression classifier
    model = LogisticRegression(max_iter=1000)
    # The .fit() command is where the actual math and learning happens
    model.fit(X_train_vectorized, Y_train)
    
# ==========================================
# 5. Evaluating the Model
# ==========================================
    print("\n[INFO] Evaluating Model Performance on Test Data:")
    
    # Ask the model to predict the answers for the 20% test data
    predictions = model.predict(X_test_vectorized)
    
    # Calculate metrics
    acc = accuracy_score(Y_test, predictions)
    prec = precision_score(Y_test, predictions)
    rec = recall_score(Y_test, predictions)
    conf_matrix = confusion_matrix(Y_test, predictions)
    
    print(f"Accuracy:  {acc * 100:.2f}% (Overall correctness)")
    print(f"Precision: {prec * 100:.2f}% (When it predicts Real, how often is it right?)")
    print(f"Recall:    {rec * 100:.2f}% (Out of all Real articles, how many did it find?)")
    print("\nConfusion Matrix:")
    print(conf_matrix)
    print("Format: [[True Negatives (Fake identified as Fake), False Positives (Fake identified as Real)]")
    print("         [False Negatives (Real identified as Fake), True Positives (Real identified as Real)]]")
    
# ==========================================
# 6. Saving the Model and Vectorizer
# ==========================================
    print("\n[INFO] Saving model and vectorizer to disk...")
    # Save the trained AI model
    with open(r'C:\Users\admin\Desktop\Fake-News-Phishing-Detector\models\fake_news_model.pkl', 'wb') as f:
        pickle.dump(model, f)
        
    # Save the TF-IDF vectorizer (Crucial! The backend needs this to understand new words)
    with open(r'C:\Users\admin\Desktop\Fake-News-Phishing-Detector\models\vectorizer.pkl', 'wb') as f:
        pickle.dump(vectorizer, f)
        
    print("[SUCCESS] Training Complete. Files saved in the 'models' directory.")

# Run the training process if the script is executed
if __name__ == "__main__":
    train_model()