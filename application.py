
from flask import Flask, render_template, request
import pickle
import pandas as pd

app = Flask(__name__)

# Load model and dataset
model = pickle.load(open('LinearRegressionModel.pkl', 'rb'))
car = pd.read_csv('Cleaned_Car_data.csv')


@app.route('/')
def index():
    companies = sorted(car['company'].dropna().unique())
    car_models = sorted(car['name'].dropna().unique())
    years = sorted(car['year'].dropna().unique(), reverse=True)
    fuel_types = sorted(car['fuel_type'].dropna().unique())

    companies.insert(0, 'Select Company')

    return render_template(
        'index.html',
        companies=companies,
        car_models=car_models,
        years=years,
        fuel_types=fuel_types
    )


@app.route('/predict', methods=['POST'])
def predict():
    try:
        company = request.form.get('company')
        car_model = request.form.get('car_models')
        year = request.form.get('year')
        fuel_type = request.form.get('fuel_type')
        driven = request.form.get('kilo_driven')

        if not all([company, car_model, year, fuel_type, driven]):
            return "Please fill all car details.", 400

        if company == 'Select Company':
            return "Please select a company.", 400

        year = int(year)
        driven = int(driven)

        if driven < 0:
            return "Kilometres cannot be negative.", 400

        input_data = pd.DataFrame([{
            'name': car_model,
            'company': company,
            'year': year,
            'kms_driven': driven,
            'fuel_type': fuel_type
        }])

        prediction = model.predict(input_data)[0]

        return str(round(float(prediction), 2))

    except ValueError:
        return "Please enter valid year and kilometres.", 400

    except Exception as error:
        print("Prediction Error:", error)
        return "Prediction error. Check terminal.", 500


if __name__ == '__main__':
    app.run(debug=True)
