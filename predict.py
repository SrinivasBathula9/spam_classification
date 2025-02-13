import joblib
from preprocess import process_email

# Load Model & Vectorizer
model = joblib.load('spam_classifier.pkl')
vectorizer = joblib.load('vectorizer.pkl')

# Predict Function
def predict_email(email_text):
    processed_email = process_email(email_text)
    email_vector = vectorizer.transform([processed_email]).toarray()
    
    prediction = model.predict(email_vector)[0]
    return "Spam" if prediction == 1 else "Not Spam"

# Test with a sample email
if __name__ == "__main__":
    test_email = input("Enter an email message: ")
    print(f"Prediction: {predict_email(test_email)}")
