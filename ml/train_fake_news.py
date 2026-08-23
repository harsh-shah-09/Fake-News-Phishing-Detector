import os
import pickle
import re
import string
import numpy as np
import pandas as pd
from scipy.sparse import hstack, csr_matrix

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import FeatureUnion
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.ensemble import ExtraTreesClassifier, StackingClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import accuracy_score, classification_report

import nltk
from nltk.corpus import stopwords

nltk.download('stopwords', quiet=True)
stop_words = set(stopwords.words('english'))

def extract_stylometric_features(raw_texts):
    """
    Extracts forensic linguistic signatures:
    1. Capital letter ratio (screaming / sensationalism)
    2. Exclamation mark density
    3. Question mark density
    4. Punctuation density
    5. Average word length
    6. Lexical diversity (unique words / total words)
    """
    features = []
    for text in raw_texts:
        text_str = str(text)
        total_chars = max(len(text_str), 1)
        words = text_str.split()
        total_words = max(len(words), 1)
        
        cap_count = sum(1 for c in text_str if c.isupper())
        excl_count = text_str.count('!')
        ques_count = text_str.count('?')
        punct_count = sum(1 for c in text_str if c in string.punctuation)
        avg_word_len = sum(len(w) for w in words) / total_words
        lexical_diversity = len(set(words)) / total_words
        
        features.append([
            cap_count / total_chars,
            excl_count / total_words,
            ques_count / total_words,
            punct_count / total_chars,
            avg_word_len,
            lexical_diversity
        ])
    return np.array(features)

def clean_text(text):
    text = str(text).lower()
    text = re.sub(r'\[.*?\]', '', text)
    text = re.sub(r'https?://\S+|www\.\S+', '', text)
    text = re.sub(r'<.*?>+', '', text)
    text = re.sub(r'[%s]' % re.escape(string.punctuation), '', text)
    text = re.sub(r'\n', ' ', text)
    text = re.sub(r'\w*\d\w*', '', text)
    words = [w for w in text.split() if w not in stop_words]
    return ' '.join(words)

def load_and_prepare_data():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    fake_df = pd.read_csv(os.path.join(base_dir, r'C:\Users\admin\Desktop\Fake-News-Phishing-Detector\datasets\Fake.csv'))
    true_df = pd.read_csv(os.path.join(base_dir, r'C:\Users\admin\Desktop\Fake-News-Phishing-Detector\datasets\True.csv'))

    # Neutralize publisher dateline leakage
    true_df['text'] = true_df['text'].str.replace(r'^.*?\(reuters\)\s*-\s*', '', regex=True, case=False)
    true_df['text'] = true_df['text'].str.replace(r'^.*?\s*-\s*', '', regex=True)

    try:
        local_df = pd.read_csv(os.path.join(base_dir, r'C:\Users\admin\Desktop\Fake-News-Phishing-Detector\datasets\..0local_real_news.csv'))
        true_df = pd.concat([true_df, local_df], axis=0)
    except FileNotFoundError:
        pass

    fake_df['label'] = 0
    true_df['label'] = 1

    df = pd.concat([fake_df, true_df], axis=0).sample(frac=1, random_state=42).reset_index(drop=True)
    df['raw_content'] = df['title'].fillna('') + ' ' + df['text'].fillna('')
    df['clean_content'] = df['raw_content'].apply(clean_text)
    
    return df['raw_content'], df['clean_content'], df['label']

def train_model():
    raw_texts, cleaned_texts, y = load_and_prepare_data()

    print("[INFO] Extracting Stylometric & Forensic Signals...")
    stylometric_matrix = extract_stylometric_features(raw_texts)

    print("[INFO] Vectorizing N-Grams (Word + Sub-word Characters)...")
    tfidf_union = FeatureUnion([
        ('word_tfidf', TfidfVectorizer(ngram_range=(1, 2), max_features=8000)),
        ('char_tfidf', TfidfVectorizer(analyzer='char', ngram_range=(3, 5), max_features=8000))
    ])
    tfidf_matrix = tfidf_union.fit_transform(cleaned_texts)

    # Fuse stylometry and TF-IDF into a single high-dimensional matrix
    X = hstack([tfidf_matrix, csr_matrix(stylometric_matrix)])

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    print("[INFO] Building 2-Tier Stacking Super-Learner...")
    base_learners = [
        ('lr', LogisticRegression(max_iter=1000, class_weight='balanced')),
        ('svc', CalibratedClassifierCV(LinearSVC(class_weight='balanced', random_state=42))),
        ('et', ExtraTreesClassifier(n_estimators=100, max_depth=15, random_state=42, n_jobs=-1))
    ]

    meta_model = StackingClassifier(
        estimators=base_learners,
        final_estimator=LogisticRegression(),
        cv=5,
        n_jobs=-1
    )

    print("[INFO] Fitting Meta-Learner (Cross-Validating Base Estimators)...")
    meta_model.fit(X_train, y_train)

    preds = meta_model.predict(X_test)
    print(f"\n[INFO] Stacking Model Accuracy: {accuracy_score(y_test, preds):.2%}")
    print("\nClassification Report:\n", classification_report(y_test, preds))

    base_dir = os.path.dirname(os.path.abspath(__file__))
    with open(os.path.join(base_dir, '../models/fake_news_model.pkl'), 'wb') as f:
        pickle.dump(meta_model, f)
    with open(os.path.join(base_dir, '../models/vectorizer.pkl'), 'wb') as f:
        pickle.dump(tfidf_union, f)
    print("[SUCCESS] Production artifacts successfully saved.")

if __name__ == '__main__':
    train_model()