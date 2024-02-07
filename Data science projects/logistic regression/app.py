import streamlit as st
import pandas as pd
from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn import linear_model
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

data = pd.read_csv('data.csv')
data['diagnosis'].replace({'M': 1, 'B': 0}, inplace=True)

data.to_csv('data.csv')

data_path = 'data.csv'
dataframe = pd.read_csv(data_path)
dataframe = dataframe[['diagnosis', 'radius_mean', 'area_mean', 'radius_se', 'area_se', 'smoothness_mean', 'smoothness_se']]

st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to", ["Home", "Visualization", "Model Stats"])

if page == "Home":
    st.title("Breast Cancer Diagnosis Predictor")

    st.subheader("Original Dataset:")
    st.write(dataframe)

elif page == "Visualization":
    st.title("Data Visualization")

    st.subheader("Scatter Plot:")
    scatter_fig, scatter_ax = plt.subplots()
    sns.catplot(x='diagnosis', y='radius_mean', data=dataframe, ax=scatter_ax)
    st.pyplot(scatter_fig)

    st.subheader("Scatter Plot with Predicted Boundaries:")
    boundary = 10
    boundary_fig, boundary_ax = plt.subplots()
    sns.scatterplot(x='radius_mean', y='diagnosis', data=dataframe, ax=boundary_ax)
    boundary_ax.plot([boundary, boundary], [0, 1], 'g', linewidth=6)
    st.pyplot(boundary_fig)

elif page == "Model Stats":
    st.title("Model Statistics")

    train_df, test_df = train_test_split(dataframe, test_size=0.4, random_state=1)

    input_labels = ['radius_mean']
    output_label = 'diagnosis'

    x_train = train_df[input_labels]
    y_train = train_df[output_label]

    class_rm = linear_model.LogisticRegression()
    class_rm.fit(x_train, y_train)

    x_test = test_df[input_labels]
    y_test = test_df[output_label].values.squeeze()
    y_pred = class_rm.predict(x_test)
    cnf_matrix = metrics.confusion_matrix(y_test, y_pred)

    st.subheader("Confusion Matrix:")
    cnf_matrix_fig, cnf_matrix_ax = plt.subplots()
    sns.heatmap(pd.DataFrame(cnf_matrix), annot=True, cmap="YlGnBu", fmt='g', ax=cnf_matrix_ax)
    cnf_matrix_ax.xaxis.set_label_position("top")
    plt.tight_layout()
    plt.ylabel('Actual diagnosis')
    plt.xlabel('Predicted diagnosis')
    st.pyplot(cnf_matrix_fig)
    

    st.subheader("Model Metrics:")
    model_stats(y_test, y_pred)

    st.subheader("Scatter Plot with Predicted Probabilities:")
    scatter_prob_fig, scatter_prob_ax = plt.subplots()
    x_test_view = x_test[input_labels].values.squeeze()
    y_prob = class_rm.predict_proba(x_test)
    sns.scatterplot(x=x_test_view, y=y_prob[:, 1], hue=y_test, ax=scatter_prob_ax)
    plt.xlabel('Radius')
    plt.ylabel('Predicted Probability')
    plt.legend()
    st.pyplot(scatter_prob_fig)
