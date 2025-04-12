import gradio as gr
from keras.models import load_model
from sklearn.feature_extraction.text import HashingVectorizer
import numpy as np

# Load the saved model
model = load_model("sentiment.h5")

# Initialize the HashingVectorizer with the same parameters used during training
vectorizer = HashingVectorizer(n_features=2**14)  # Use the same n_features as in training

# Prediction function
def predict_sentiment(text):
    # Transform the input text into the hashed feature space
    new_test = vectorizer.transform([text])
    
    # Predict the sentiment of the review
    res = np.argmax(model.predict(new_test))
    
    # Output the result
    return "Positive 😊" if res == 1 else "Negative 😞"

# Create Gradio UI
iface = gr.Interface(
    fn=predict_sentiment,
    inputs=gr.Textbox(lines=3, placeholder="Type a sentence here..."),
    outputs="text",
    title="Sentiment Analyzer",
    description="Enter a sentence to see if it's positive or negative.",
)

# Launch the app
iface.launch()
