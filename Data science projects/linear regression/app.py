import streamlit as st
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
import seaborn as sns
import matplotlib.pyplot as plt

dataset = pd.read_csv('50_Startups.xlsx')
X = dataset.iloc[:, :-1]
y = dataset.iloc[:, 4]

states = pd.get_dummies(X['State'], drop_first=True)
X = X.drop('State', axis=1)
X = pd.concat([X, states], axis=1)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

regressor = LinearRegression()
regressor.fit(X_train, y_train)
y_pred = regressor.predict(X_test)

score = r2_score(y_test, y_pred)

st.title('Linear Regression App')

menu = ['Home', 'R-squared Score', 'Visualizations']
choice = st.sidebar.selectbox('Select Page', menu)

if choice == 'Home':
    st.subheader('Original Dataset:')
    st.write(dataset)

elif choice == 'R-squared Score':
    st.subheader('R-squared Score:')
    st.write(f'R-squared Score: {score}')

elif choice == 'Visualizations':
    st.subheader('Actual vs Predicted Values with Linear Regression Line:')
    result_df = pd.DataFrame({'Actual': y_test, 'Predicted': y_pred})
    result_df = result_df.sort_values(by='Actual')

    fig, ax = plt.subplots()
    sns.scatterplot(x='Actual', y='Predicted', data=result_df, color='blue', label='Actual vs Predicted', ax=ax)
    sns.regplot(x='Actual', y='Predicted', data=result_df, ci=None, color='red', label='Linear Regression Line', ax=ax)
    st.pyplot(fig)

    st.subheader('Actual vs Predicted Values:')
    result_df = result_df.sort_values(by='Actual')

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(result_df['Actual'].values, label='Actual', marker='o', linestyle='-')
    ax.plot(result_df['Predicted'].values, label='Predicted', marker='o', linestyle='--')
    st.pyplot(fig)
