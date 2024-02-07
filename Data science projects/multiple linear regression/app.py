# app.py
import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
import matplotlib.pyplot as plt

dataset_path = "insurance.csv"
df = pd.read_csv(dataset_path)

def train_model(df):
    df['sex'] = df['sex'].astype('category').cat.codes
    df['smoker'] = df['smoker'].astype('category').cat.codes
    df['region'] = df['region'].astype('category').cat.codes

    X = df.drop(columns="charges")
    y = df['charges']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)

    lr = LinearRegression()
    lr.fit(X_train, y_train)

    return X_train, X_test, y_train, y_test, lr

def plot_scatter(x_actual, y_actual, x_predicted, y_predicted):
    fig, ax = plt.subplots()
    ax.scatter(x_actual, y_actual, label='Actual charges')
    ax.scatter(x_predicted, y_predicted, label='Predicted charges')
    ax.set_xlabel("Charges")
    ax.set_ylabel("Predicted Charges")
    ax.set_title("Actual vs Predicted charges")
    ax.legend()
    st.pyplot(fig)

def display_r2_score(actual, predicted):
    r2 = r2_score(actual, predicted)
    st.write(f"R-squared Score: {r2}")

def page1():
    st.title("Page 1: Dataset Overview")
    st.dataframe(df.head())

def page2():
    st.title("Page 2: Model Training and Results")

    X_train, X_test, y_train, y_test, lr = train_model(df)

    y_pred_train = lr.predict(X_train)
    y_pred_test = lr.predict(X_test)

    st.subheader("Training Set")
    plot_scatter(y_train, y_pred_train, X_train, X_train)

    st.subheader("Test Set")
    plot_scatter(y_test, y_pred_test, X_test, X_test)

    combined_actual = list(y_train) + list(y_test)
    combined_predicted = list(y_pred_train) + list(y_pred_test)

    st.subheader("Combined Set")
    plot_scatter(combined_actual, combined_predicted, combined_actual, combined_predicted)

    display_r2_score(combined_actual, combined_predicted)

def main():
    st.sidebar.title("Navigation")
    selection = st.sidebar.radio("Go to", ["Dataset Overview", "Model Training and Results"])

    if selection == "Dataset Overview":
        page1()
    elif selection == "Model Training and Results":
        page2()

if __name__ == "__main__":
    main()
