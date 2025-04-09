from flask import Flask, render_template, url_for, request
import numpy as np
import tensorflow as tf
from tensorflow import keras
from keras.preprocessing.text import Tokenizer
import joblib
# from keras.preprocessing.sequence import pad_sequences
import pickle
import re

model = tf.keras.models.load_model("models/sentiment_analysis_model.keras")
tokenizer = joblib.load("models/tokenizer.pkl")

max_length = 300

from tensorflow.keras.preprocessing.sequence import pad_sequences

def predict_sentiment(text):
    sequence = tokenizer.texts_to_sequences([text])  # Convert to sequences
    sequence_padded = pad_sequences(sequence, maxlen=max_length, padding="post")

    prediction = model.predict(sequence_padded)[0][0]  # Get probability

    return "Positive" if prediction > 0.5 else "Negative"

app = Flask(__name__)
 
@app.route("/", methods=["GET","POST"])
def home():
	return(render_template("home.html"))

@app.route("/predict", methods=["GET", "POST"])
def predict():
    if request.method == "POST":
        user_input = request.form['text']
        prediction = predict_sentiment(user_input)
        return render_template("home.html", prediction=prediction)
    return render_template("home.html", prediction=None)
	
if __name__ == "__main__":
	app.run()
