from flask import Flask, render_template, request
import pandas as pd
import numpy as np
import pickle

import os

#with open(os.path.join('/media/yong/storage/ds_3/deployment', 'Procfile'), "w") as file1:
#    toFile = 'web: gunicorn app:app'

#file1.write(toFile)

app = Flask(__name__)

#load the saved ml model
def load_model():
    #return pickle.load('./model/orf.pkl', 'rb')
    #return pickle.load('orf.pkl', 'rb')
    return pickle.load('orf.pkl', 'rb')
@app.route('/')
def home():
    return render_template('index.html')

#predict the result and return it
@app.route('/predict', methods=['POST'])
def predict():
    '''
    For rendering results on HTML GUI
    '''

    labels = ['Rejection', 'Approval']

    #text1 = request.form['Applicant Income']
    #text2 = request.form['Coapplicant Income']
    #text3 = request.form['Loan Amount']
    #text4 = request.form['Credit History']


    features = [float(x) for x in request.form.values()]
    values = [np.array(features)]
    model = load_model()
    prediction = model.predict(values)

    result = labels[prediction[0]]

    return render_template('index.html', output='Your application is likely get {}'.format(result))
if __name__ == '__main__':
    #port=int(os.environ.get('PORT', 5000))
    app.run(debug=True)
