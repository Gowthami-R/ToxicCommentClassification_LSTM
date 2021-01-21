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


f1 = open('./templates/preds.html', 'w')
f2 = open('./templates/index.html', 'w')
f1.write(preds)
f2.write(index)
f1.close()
f2.close()

app.run()