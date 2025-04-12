# Load the necessary libraries
from keras.models import load_model
from sklearn.feature_extraction.text import HashingVectorizer
import numpy as np

# Load the saved model
model = load_model("sentiment.h5")

# Initialize the HashingVectorizer with the same parameters used during training
vectorizer = HashingVectorizer(n_features=2**20)  # Use the same n_features as in training

# Inference: Taking user input for review
ww = input("Enter your review: ")

# Create the test set with user input
test_set = [ww]

# Transform the input text into the hashed feature space (same way it was done during training)
new_test = vectorizer.transform(test_set)

# Predict the sentiment of the review
res = np.argmax(model.predict(new_test))

# Output the result
if res == 0:
    print("Negative")
else:
    print("Positive")
