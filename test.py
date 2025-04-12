import tkinter as tk
from tkinter import messagebox
from keras.models import load_model
import numpy as np
from sklearn.feature_extraction.text import HashingVectorizer

# Load the trained model
model = load_model("sentiment.h5")

# Initialize the HashingVectorizer with the same number of features as used during training
vectorizer = HashingVectorizer(n_features=2**20)  # Same number of features used in training

# Function to predict sentiment
def predict_sentiment():
    review = entry.get()  # Get user input
    if not review.strip():
        messagebox.showerror("Error", "Please enter a review.")
        return

    # Transform input using the loaded vectorizer (without needing to load a file)
    new_test = vectorizer.transform([review])

    # Predict sentiment
    res = np.argmax(model.predict(new_test))

    # Show result
    sentiment = "Positive 😊" if res == 1 else "Negative 😞"
    result_label.config(text=f"Sentiment: {sentiment}")

# Create GUI window
root = tk.Tk()
root.title("Sentiment Analyzer")

# Create and place widgets
tk.Label(root, text="Enter your review:").pack(pady=5)
entry = tk.Entry(root, width=50)
entry.pack(pady=5)

submit_btn = tk.Button(root, text="Submit", command=predict_sentiment)
submit_btn.pack(pady=5)

result_label = tk.Label(root, text="", font=("Arial", 12, "bold"))
result_label.pack(pady=10)

# Run the GUI
root.mainloop()
