import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix,ConfusionMatrixDisplay
import joblib

df=pd.read_csv('sonar.csv')
X=df.iloc[:,:-1]
y=df.iloc[:,-1]
y=np.where(y=='R', 0,1)

X = (X - np.mean(X, axis=0)) / np.std(X, axis=0)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


def sigmoid(z):
    return 1 / (1 + np.exp(-z))


def compute_cost(X, y, weights, bias, lambda_):
    m = X.shape[0]
    z = np.dot(X, weights) + bias
    h = sigmoid(z)
    reg_term = (lambda_ / (2 * m)) * np.sum(weights ** 2)
    cost = - (1/m) * np.sum(y * np.log(h + 1e-8) + (1 - y) * np.log(1 - h + 1e-8)) + reg_term
    return cost


def gradient_descent(X, y, weights, bias, lr, iterations, lambda_):
    m = X.shape[0]
    cost_history = []

    for i in range(iterations):
        z = np.dot(X, weights) + bias
        h = sigmoid(z)

        dw = (1/m) * np.dot(X.T, (h - y)) + (lambda_ / m) * weights
        db = (1/m) * np.sum(h - y)

        weights -= lr * dw
        bias -= lr * db

        if i % 100 == 0:
            cost = compute_cost(X, y, weights, bias, lambda_)
            cost_history.append(cost)
            print(f"Iteration {i}, Cost: {cost:.4f}")

    return weights, bias, cost_history


weights = np.zeros(X_train.shape[1])
bias = 0
lambda_ = 0.1
weights, bias, cost_history = gradient_descent(X_train, y_train, weights, bias, lr=0.01, iterations=100, lambda_=lambda_)

# Predict
def predict(X, weights, bias):
    probs = sigmoid(np.dot(X, weights) + bias)
    return np.where(probs >= 0.5, 1, 0)

# Evaluate
y_pred = predict(X_test, weights, bias)
accuracy = np.mean(y_pred == y_test)
print(f"Accuracy: {accuracy * 100:.2f}%")

y_pred = predict(X_test, weights, bias)
cm = confusion_matrix(y_test, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm)
disp.plot(cmap='pink')
plt.title('Confusion Matrix')
plt.savefig("confusion_matrix.png", dpi=300, bbox_inches='tight')
plt.show()


param={
    "weights" : weights,
    "bias" : bias,
    "mean" : np.mean(X,axis=0),
    "std" : np.std(X,axis=0)
}
joblib.dump(param,"sonar_vs_rock.pkl")


