import numpy as np
from flask import Flask, render_template, url_for, request     
from model_files.ml_model import ToxicClassifier


app = Flask(__name__)

@app.route('/')
def input():
  return render_template('index.html')

@app.route('/predictions', methods=["GET", "POST"])
def preds():
  if request.method == "POST":
    text = request.form.get('text')
    tcc = ToxicClassifier()
    processed = tcc.preprocess(text)
    model = tcc.get_model()
    pred = model.predict(processed)[0]
    return render_template('preds.html', p = pred, text = text)

app.run()