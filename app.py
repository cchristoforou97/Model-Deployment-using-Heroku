import pickle
from flask import Flask, request, render_template
import numpy as np

app = Flask(__name__)
model = pickle.load(open('model.pkl', 'rb'))

@app.route('/')
def home():
	return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
	input = [float(x) for x in request.form.values()]
	input = [np.array(input)]

	prediction = model.predict(input).item()

	return render_template('index.html', prediction_text = 'The species of this flower is {}'.format(prediction))

if __name__ == '__main__':
	app.run(debug=True)
