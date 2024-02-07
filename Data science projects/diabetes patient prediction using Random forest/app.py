import streamlit as st
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn import metrics
from sklearn.metrics import confusion_matrix, classification_report

sns.set()

@st.cache_data
def load_data():
    return pd.read_csv('diabetes.csv')

def main():
    st.title("Diabetes Prediction Web App")

    page = st.sidebar.selectbox("Select Page", ["Home", "Explore Data", "Train Model"])

    if page == "Home":
        home_page()
    elif page == "Explore Data":
        explore_data_page()
    elif page == "Train Model":
        train_model_page()

def home_page():
    st.header("Welcome to the Home Page!")
    st.write("This is the landing page of your application.")

def explore_data_page():
    st.header("Explore Data Page")
    diabetes_df = load_data()
    st.dataframe(diabetes_df.head())

def train_model_page():
    st.header("Train Model Page")

    diabetes_df = load_data()

    st.subheader("Standardized Features (First 5 rows)")
    sc_X = StandardScaler()
    X = pd.DataFrame(sc_X.fit_transform(diabetes_df.drop(["Outcome"], axis=1)),
                     columns=['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI',
                              'DiabetesPedigreeFunction', 'Age'])
    st.dataframe(X.head())

    test_size = st.slider("Select Test Size:", 0.1, 0.5, 0.33, step=0.05)

    X_train, X_test, y_train, y_test = train_test_split(diabetes_df.drop('Outcome', axis=1), diabetes_df['Outcome'],
                                                        test_size=test_size, random_state=7)

    rfc = RandomForestClassifier(n_estimators=200)
    rfc.fit(X_train, y_train)

    rfc_train = rfc.predict(X_train)
    st.write("Train Accuracy:", format(metrics.accuracy_score(y_train, rfc_train)))

    predictions = rfc.predict(X_test)
    st.write("Test Accuracy:", format(metrics.accuracy_score(y_test, predictions)))

    st.subheader("Confusion Matrix")
    st.write(confusion_matrix(y_test, predictions))

    st.subheader("Classification Report")
    st.text(classification_report(y_test, predictions))

if __name__ == "__main__":
    main()
