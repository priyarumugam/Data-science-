import streamlit as st
from sklearn.datasets import load_digits
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
import seaborn as sns

digits = load_digits()

normalized_images = digits.images / 16.0

st.title("Digits Classification App")

st.subheader("Sample Digits:")
for i in range(5):
    st.image(normalized_images[i], caption=f"Digit {i}", use_column_width=True)

st.subheader("Model Training:")
model = LogisticRegression()
X_train, X_test, y_train, y_test = train_test_split(digits.data, digits.target, test_size=0.2)
model.fit(X_train, y_train)
accuracy = model.score(X_test, y_test)
st.write(f"Model Accuracy: {accuracy:.2f}")

st.subheader("Predictions and Confusion Matrix:")
y_predicted = model.predict(X_test)
cm = confusion_matrix(y_test, y_predicted)

fig, ax = plt.subplots(figsize=(10, 7))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax)
plt.xlabel('Predicted')
plt.ylabel('Truth')
st.pyplot(fig)

st.subheader("Predictions:")
sample_predictions = model.predict(digits.data[0:5])
st.write(f"Predictions for the first 5 digits: {sample_predictions}")

