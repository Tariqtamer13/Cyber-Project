import streamlit as st
import pickle
import numpy as np
from feature import FeatureExtraction


def load_model():
    with open("models/Phishing URL Detection model.pkl", "rb") as file:
        return pickle.load(file)


# Load the trained model
gbc = load_model()

# App title and description
st.title("Phishing URL Detection 🚨")
st.write("Enter a URL below to check if it's phishing or safe.")

# URL input
url = st.text_input("URL to check:", placeholder="https://example.com")

# Check button
if st.button("Check URL"):
    if url:
        try:
            # Feature extraction
            obj = FeatureExtraction(url)
            features = np.array(obj.getFeaturesList()).reshape(1, -1)

            # Predictions
            y_pred = gbc.predict(features)[0]
            y_pro_phishing = gbc.predict_proba(features)[0][0]
            y_pro_non_phishing = gbc.predict_proba(features)[0][1]

            # Display results
            if y_pred == 1:
                st.success(f"✅ Safe to go: {y_pro_non_phishing * 100:.2f}% confidence")
            else:
                st.error(
                    f"⚠️ Potential phishing: {y_pro_phishing * 100:.2f}% confidence"
                )

            # Probabilities breakdown
            st.write(f"Phishing Probability: {y_pro_phishing * 100:.2f}%")
            st.write(f"Non-Phishing Probability: {y_pro_non_phishing * 100:.2f}%")
        except Exception as e:
            st.error(f"Error processing URL: {e}")
    else:
        st.warning("Please enter a valid URL.")

# Footer
st.write("---")
st.write("*Powered by Streamlit & Gradient Boosting Classifier*")
