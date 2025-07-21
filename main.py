
from click.core import batch
from flask import Flask, render_template
import pandas as pd
import numpy as np
import pickle

app = Flask(__name__)
data = pd.read_csv("Cleaned_dataset.csv")
pipe = pickle.load(open("RidgeModel.pkl",'rb'))

@app.route("/")
def home():
    locations = sorted(data['location'].unique())
    return render_template('index.html', locations=locations)
from flask import request, jsonify

@app.route('/predict', methods=['POST'])
def predict():
    location = request.form.get('location')
    bhk = request.form.get('bhk')
    bath = request.form.get('bath')
    sqft = request.form.get('total_sqft')

    print(location, bhk, bath, sqft)

    input_df = pd.DataFrame([[location, float(sqft), float(bath), int(bhk)]],
                            columns=['location', 'total_sqft', 'bath', 'bhk'])

    prediction = pipe.predict(input_df)[0]* 1e5

    return str(np.round(prediction, 2))




import os

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 10000))  # Render assigns this dynamically
    app.run(debug=True, host='0.0.0.0', port=port)  # ✅ MUST use 0.0.0.0


