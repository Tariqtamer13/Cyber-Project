import pickle
import numpy as np
from feature import FeatureExtraction

print("Loading model...")

with open("models/Phishing URL Detection model.pkl", "rb") as file:
    gbc = pickle.load(file)

print("Model loaded successfully!")
print("Model is ready to use!")
print("Please enter the URL to check if it is phishing or not.")

url = input("Enter the URL to check: ")
obj = FeatureExtraction(url)

print("Extracting features...")
x = np.array(obj.getFeaturesList()).reshape(1, 30)

y_pred = gbc.predict(x)[0]
y_pro_phishing = gbc.predict_proba(x)[0, 0]
y_pro_non_phishing = gbc.predict_proba(x)[0, 1]

if y_pred == 1:
    pred = "It is {0:.2f} % safe to go ".format(y_pro_phishing * 100)
else:
    pred = "It is {0:.2f} % not safe to go ".format(y_pro_non_phishing * 100)

print("Prediction: ", pred)
print("Phishing Probability: ", y_pro_phishing * 100)
print("Non-Phishing Probability: ", y_pro_non_phishing * 100)
print("Thank you for using the Phishing URL Detection model!")
