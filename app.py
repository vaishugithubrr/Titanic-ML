import pickle
import pandas as pd
from flask import Flask, render_template, request

app = Flask(__name__)

# Load the best SVC model
with open('best_svc_save (1).pkl', 'rb') as file:
    best_svc_model = pickle.load(file)

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get data from HTML form
        pclass = float(request.form['pclass'])
        sex = float(request.form['sex'])
        age = float(request.form['age'])
        fare = float(request.form['fare'])
        cabin = float(request.form['cabin'])
        embarked = float(request.form['embarked'])
        title = float(request.form['title'])
        famsize = float(request.form['famsize'])

        # Create a DataFrame with the user input
        input_data = pd.DataFrame({
            'Pclass': [pclass],
            'Sex': [sex],
            'Age': [age],
            'Fare': [fare],
            'Cabin': [cabin],
            'Embarked': [embarked],
            'Title': [title],
            'FamilySize': [famsize]
        })

        # Make prediction using the loaded model
        prediction = best_svc_model.predict(input_data)[0]

        # Process prediction result
        if prediction == 1:
            result = 'Survived'
        else:
            result = 'Not Survived'

        return render_template("home.html", prediction_result="Prediction is : {}".format(result))

    except Exception as e:
        return render_template("home.html", prediction_result="Error: {}".format(str(e)))

if __name__ == "__main__":
    app.run(debug=True)