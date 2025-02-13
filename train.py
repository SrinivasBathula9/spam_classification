from preprocess import load_and_preprocess_data, vectorize_data, split_data
from model import train_model
from evaluate import evaluate_model

# Load and preprocess data
df = load_and_preprocess_data('spam.csv')

# Vectorize
X, y = vectorize_data(df)

# Split into Train & Test
X_train, X_test, y_train, y_test = split_data(X, y)

# Train Model
model = train_model(X_train, y_train)

# Evaluate Model
evaluate_model(model, X_test, y_test)
