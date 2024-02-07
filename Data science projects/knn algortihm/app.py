import streamlit as st
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns

iris = load_iris()
df = pd.DataFrame(iris.data, columns=iris.feature_names)
df['target'] = iris.target
df['flower_name'] = df.target.apply(lambda x: iris.target_names[x])

page = st.sidebar.selectbox("Select Page", ["Home", "KNN Classification"])
selected_feature = None
if page == "Home":
    st.title("Iris Dataset - Home")
    st.write("Welcome to the Home page!")
elif page == "KNN Classification":
    st.title("Iris Dataset - KNN Classification")
    
    selected_feature = st.sidebar.selectbox("Select Feature", df.columns[:-2])
    
    st.write("### Sample Data:")
    st.write(df.head())

    X = df.drop(['target', 'flower_name'], axis='columns')
    y = df.target
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)

    k_value = st.sidebar.slider("Select the number of neighbors (k)", min_value=1, max_value=20, value=5)
    knn = KNeighborsClassifier(n_neighbors=k_value)
    knn.fit(X_train, y_train)

    accuracy = knn.score(X_test, y_test)
    st.write(f"### K-Nearest Neighbors (KNN) Classifier (k={k_value})")
    st.write(f"Accuracy: {accuracy:.2f}")

    st.write("### Confusion Matrix:")
    y_pred = knn.predict(X_test)
    cm = confusion_matrix(y_test, y_pred)

    fig, ax = plt.subplots()
    ax = sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False)
    plt.xlabel('Predicted')
    plt.ylabel('Truth')
    st.pyplot(fig)

    st.write("### Classification Report:")
    classification_rep = classification_report(y_test, y_pred)
    st.text(classification_rep)

if selected_feature:
    st.write(f"Selected Feature: {selected_feature}")
