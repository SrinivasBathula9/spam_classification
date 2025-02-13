from sklearn.metrics import accuracy_score, f1_score
import joblib

# Load Model
def load_model():
    return joblib.load('spam_classifier.pkl')

# Evaluate Model
def evaluate_model(model, X_test, y_test):
    y_pred = model.predict(X_test)
    
    accuracy = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    
    print(f"Accuracy: {accuracy:.4f}")
    print(f"F1 Score: {f1:.4f}")
    
    return accuracy, f1
