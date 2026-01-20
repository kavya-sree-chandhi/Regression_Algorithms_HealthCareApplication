
import streamlit as st
import numpy as np
import pandas as pd
from sklearn.datasets import load_diabetes
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

st.title("🏥 Healthcare Disease Progression Predictor")

data = load_diabetes(as_frame=True)
df = data.frame

X = df.drop('target', axis=1)
y = df['target']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

model = LinearRegression()
model.fit(X_train, y_train)

st.sidebar.header("Enter Patient Data")
inputs = []
for col in X.columns:
    val = st.sidebar.slider(col, float(X[col].min()), float(X[col].max()), float(X[col].mean()))
    inputs.append(val)

input_array = scaler.transform([inputs])
prediction = model.predict(input_array)

st.subheader("Predicted Disease Progression Score:")
st.success(round(prediction[0], 2))
