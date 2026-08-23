# 🛡️ AI Fraud Detector: Fake News & Phishing

## 📖 1. Project Overview
The **AI Fraud Detector** is a full-stack Machine Learning application built to protect users from modern digital deception: misinformation and malicious phishing links. 

By combining **Natural Language Processing (NLP)**, **Forensic Stylometry**, **Character-Level Lexical Analysis**, and **Dynamic Domain Authority Guardrails (Tranco Top 1M)**, this system delivers enterprise-grade detection accuracy while remaining lightweight for cloud deployment.

---

## ⚙️ 2. Technology Stack
* **Frontend:** HTML5, CSS3, JavaScript (ES6+ Fetch API), Bootstrap 5, Chart.js
* **Backend:** Python 3.8+, Flask, Flask-CORS, Gunicorn
* **Machine Learning & NLP:** scikit-learn, NLTK, scipy, pandas, NumPy
* **Security & Verification:** Tranco (Top 1 Million Global Domain Authority API)
* **Deployment:** Vercel (Frontend UI), Render (Backend REST API)

---

## 🧠 3. Advanced Machine Learning Architecture

### A. Phishing Detection Engine (94.5%+ Accuracy)
* **Lexical Character N-Grams:** Extracts sub-word character patterns (2–5 character n-grams) via `TfidfVectorizer` to identify deceptive typosquatting, brand spoofing (e.g., `paypa1`), and parameter manipulation.
* **Logistic Regression Classifier:** Trained on balanced multi-class URL datasets with high convergence parameters (`max_iter=1000`).
* **Tranco Top 1M Guardrail:** Real-time domain authority checks against the top 1,000,000 global domains to eliminate false positives on trusted enterprise and regional platforms.

### B. Fake News Detection Engine
* **Data Leakage Mitigation:** Preprocessing pipeline strips publisher datelines (e.g., `Reuters`) to prevent shortcut learning.
* **Hybrid Feature Union:** Merges word-level n-grams (1–2 words) and character-level n-grams (3–5 characters).
* **Forensic Stylometry:** Evaluates structural linguistic markers, including capitalization ratios, punctuation spikes, lexical diversity (Type-Token Ratio), and token complexity.

---

## 🚀 4. Installation Guide (Local Development)

### Prerequisites
* Python 3.8 or higher
* Git

### Step-by-Step Setup

1. **Clone the Repository**
   ```bash
   git clone [https://github.com/YourUsername/Fake-News-Phishing-Detector.git](https://github.com/YourUsername/Fake-News-Phishing-Detector.git)
   cd Fake-News-Phishing-Detector


2. **Create a Virtual Environment**
```bash
python -m venv venv
# On Windows (cmd/PowerShell):
venv\Scripts\activate
# On macOS/Linux:
source venv/bin/activate
```


3. **Install Dependencies**
```bash
pip install -r backend/requirements.txt

```


4. **Train the Models (Mandatory before starting server)**
```bash
cd ml
python train_fake_news.py
python train_phishing.py
cd ..
```


5. **Start the Flask Backend**
```bash
cd backend
python app.py
```


6. **Launch the Frontend**
Open `frontend/index.html` in your web browser.

---

## 🖥️ 4. User Guide

1. Open the web interface.
2. Toggle between the Fake News and Phishing URL tabs.
3. **Analyze News:** Paste full article text into the input field and click **Analyze Text**.
4. **Scan URLs:** Paste any target link (e.g., https://example.com) and click **Scan Link**.
5. The application outputs an instant verdict (**Real/Safe** or **Fake/Phishing**), a confidence percentage, and an interactive Chart.js doughnut diagram showing the margin of certainty.

---

## 📡 5. API Documentation

The backend exposes two REST API endpoints. Both accept `POST` requests and return `JSON`.

### A. Predict Fake News

* **URL:** `/api/predict/news`
* **Method:** `POST`
* **Request Body:**
```json
{
  "text": "The US Supreme Court delivered a landmark trade ruling on international tariffs..."
}
```


* **Success Response (200 OK):**
```json
{
  "prediction": "Real",
  "confidence": 81.11
}

```


* **Error Response (400 Bad Request):**
```json
{
  "error": "Missing 'text' key in JSON payload"
}

```



### B. Predict Phishing URL

* **URL:** `/api/predict/url`
* **Method:** `POST`
* **Request Body:**
```json
{
  "url": "[https://gemini.google.com/app](https://gemini.google.com/app)"
}

```


* **Success Response (200 OK):**
```json
{
  "prediction": "Safe",
  "confidence": 98.50
}

```



---

## 🛠️ 6. Developer Guide

### System Architecture Breakdown

Fake-News-Phishing-Detector/
├── backend/
│   ├── app.py                # Flask application entry point
│   ├── routes.py             # API route controllers
│   ├── utils.py              # ML inference and Tranco guardrail middleware
│   ├── requirements.txt      # Production dependencies
│   └── test_api.py           # Automated unit and integration tests
├── datasets/                 # Training datasets (CSV format)
├── frontend/                 # Client-side UI
│   ├── index.html            # Main interface
│   ├── css/                  # Styling & responsiveness
│   └── js/                   # Fetch API & Chart.js logic
├── ml/                       # Model training scripts
│   ├── train_fake_news.py    # NLP training pipeline
│   └── train_phishing.py     # TF-IDF lexical training pipeline
└── models/                   # Serialized pickle binaries (.pkl)

### Contribution Rules

1. **Updating AI Models:** If you alter the ML algorithms or utilize larger Kaggle datasets, you must completely re-run the scripts in the `ml/` folder to overwrite the previous `.pkl` files.
2. **Continuous Integration Testing:** Always validate changes to the routing engine by running the integration tests before executing a Git commit.
```bash
cd backend
python test_api.py
