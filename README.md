# 📧 Spam Classifier

A **Spam Classifier** using **Support Vector Machine (SVM)** and **TF-IDF Vectorization** to identify spam emails. This project follows a **modular approach**, making it easy to train, test, and use the model.

---
## 🚀 Features
- **Preprocessing Module**: Cleans and tokenizes text
- **TF-IDF Vectorization**: Efficient feature extraction
- **Support Vector Machine (SVM) Model**: Trained for spam detection
- **Model Persistence**: Saves model and vectorizer for reuse
- **User-Friendly CLI**: Easily test emails for spam classification

---
## 📂 Project Structure
```
spam_classifier/
│── preprocess.py        # Data preprocessing (cleaning, vectorization)
│── model.py            # Train and save the model
│── evaluate.py         # Model evaluation metrics
│── predict.py          # User input prediction script
│── train.py            # Main training pipeline
│── spam.csv            # Dataset file
│── README.md           # Project documentation
```

---
## 🔧 Setup Instructions
### 1️⃣ Install Dependencies
Ensure you have Python installed. Then, install the required libraries:
```bash
pip install pandas numpy scikit-learn nltk joblib
```

### 2️⃣ Download NLTK Data
```python
import nltk
nltk.download('punkt')
```

### 3️⃣ Run Training Pipeline
Train the model using:
```bash
python train.py
```
This will:
- Load and preprocess the dataset
- Train an **SVM classifier**
- Save the trained model (`spam_classifier.pkl`)
- Save the **TF-IDF vectorizer** (`vectorizer.pkl`)

### 4️⃣ Test Model with New Emails
Run the prediction script:
```bash
python predict.py
```
Enter a sample email text, and it will classify whether it is **Spam** or **Not Spam**.

---
## 🏗️ Code Modules Overview
### `preprocess.py`
- Loads and preprocesses dataset (`spam.csv`)
- Performs **text cleaning** and **stemming**
- Converts text into a **TF-IDF feature matrix**
- Saves the vectorizer (`vectorizer.pkl`)

### `model.py`
- Trains **SVM (Support Vector Machine)** model
- Saves the trained model (`spam_classifier.pkl`)

### `evaluate.py`
- Loads the model and evaluates it
- Calculates **Accuracy** and **F1 Score**

### `predict.py`
- Loads the trained model and vectorizer
- Predicts spam/not spam for user-inputted emails

### `train.py`
- Calls all necessary functions to train and evaluate the model

---
## 📊 Model Performance
- **Accuracy:** ~97%  
- **F1 Score:** ~0.96  
(Scores may vary slightly depending on dataset split)

---
## 📝 Future Improvements
- Implement **Deep Learning** for improved performance
- Deploy as a **web API** using Flask/Django
- Integrate with an **email service provider** for real-time classification

---
## 🏆 Contribution
Feel free to submit **issues** or **pull requests** to improve the project!

---
## 📜 License
This project is open-source under the **MIT License**.

