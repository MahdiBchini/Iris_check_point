import streamlit as st
from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier

# Load the iris dataset
iris = load_iris()
X = iris.data
Y = iris.target

# Create a Random Forest Classifier
clf = RandomForestClassifier()

# Fit the model
clf.fit(X, Y)

# Streamlit app
st.title("Iris Flower Type Prediction")
st.header("Enter Sepal and Petal Measurements")

# Input fields for sepal and petal measurements
sepal_length = st.slider("Sepal Length", float(X[:, 0].min()), float(X[:, 0].max()), float(X[:, 0].mean()))
sepal_width = st.slider("Sepal Width", float(X[:, 1].min()), float(X[:, 1].max()), float(X[:, 1].mean()))
petal_length = st.slider("Petal Length", float(X[:, 2].min()), float(X[:, 2].max()), float(X[:, 2].mean()))
petal_width = st.slider("Petal Width", float(X[:, 3].min()), float(X[:, 3].max()), float(X[:, 3].mean()))

# Prediction button
if st.button("Predict"):
    # Create input data for prediction
    input_data = [[sepal_length, sepal_width, petal_length, petal_width]]

    # Make a prediction
    prediction = clf.predict(input_data)[0]

    # Get the corresponding class label
    predicted_class = iris.target_names[prediction]

    # Display the prediction
    st.write(f"Predicted Iris Flower Type: {predicted_class}")

# To deploy the Streamlit app with Streamlit Share, save it as a .py file and deploy using Streamlit Share:
# streamlit run iris_classifier_app.py

