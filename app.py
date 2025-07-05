from flask import Flask, render_template, request 
import numpy as np
import joblib

model = joblib.load("sonar_vs_rock.pkl")
weights = model["weights"]
bias = model["bias"]
mean = model["mean"]
std = model["std"]

app = Flask(__name__)

def sigmoid(z):
    return 1 / (1 + np.exp(-z))


@app.route('/', methods=['GET'])
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    text = request.form['features']
    try:
        values = list(map(float, text.strip().split()))
        if len(values) != 60:
            return render_template('index.html', prediction="❌ Please enter exactly 60 values!")
    except ValueError:
        return render_template('index.html', prediction="❌ Invalid input. Please enter numbers only.")

    features = np.array(values)
    features = (features - mean) / std

    z = np.dot(features, weights) + bias
    prob = sigmoid(z)
    pred_class = "It is mine" if prob >= 0.5 else "It is rock"

    return render_template('index.html', prediction=f"{pred_class} (Probability: {prob:.2f})")

if __name__ == '__main__':
    app.run(debug=True)
