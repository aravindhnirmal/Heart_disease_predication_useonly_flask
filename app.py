from flask import Flask, render_template, request
import pickle
import numpy as np


app = Flask(__name__)

# load the trained logistic regression model
with open('lrmodel.pkl', 'rb') as file:
    model = pickle.load(file)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    # get the form data submitted by the user
    age = float(request.form['age'])
    sex = float(request.form['sex'])
    cp = float(request.form['cp'])
    trestbps = float(request.form['trestbps'])
    chol = float(request.form['chol'])
    fbs = float(request.form['fbs'])
    restecg = float(request.form['restecg'])
    thalach = float(request.form['thalach'])
    exang = float(request.form['exang'])
    oldpeak = float(request.form['oldpeak'])
    slope = float(request.form['slope'])
    ca = float(request.form['ca'])
    thal = float(request.form['thal'])

    # create a numpy array of the user input data
    input_data = np.array([[age, sex, cp, trestbps, chol, fbs, restecg, thalach, exang, oldpeak, slope, ca, thal]])

    # make a prediction on the input data
    prediction = model.predict(input_data)

    # get the prediction label (0 or 1) and convert it to a string
    if prediction[0] == 0:
        prediction_label = 'No Heart Disease'
    else:
        prediction_label = 'Heart Disease'

    # render the predict.html template and pass in the prediction result
    return render_template('predict.html', prediction=prediction_label)

if __name__ == '__main__':
    app.run(debug=True)
