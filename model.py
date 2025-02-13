from sklearn.svm import SVC
import joblib

# Train SVM Model
def train_model(X_train, y_train):
    model = SVC(kernel='linear', C=1.0)
    model.fit(X_train, y_train)
    
    joblib.dump(model, 'spam_classifier.pkl')  # Save model
    return model
