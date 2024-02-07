import streamlit as st
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import scipy.cluster.hierarchy as shc
from sklearn.cluster import AgglomerativeClustering
import os

def load_data():
    dataset_path = "C:\\Users\\Priya\\Desktop\\projects on algorithms\\hierarical\\hierarchical-clustering-with-python-and-scikit-learn-shopping-data (1).csv"
    df = pd.read_csv(dataset_path)
    return df

def overview_page(df):
    st.title("Customer Data Exploration - Overview")

    st.write("## Customer Data Overview")
    st.dataframe(df.head())

    st.write("## Missing Values")
    st.write(df.isnull().sum())

    st.write("## Dataset Information")
    st.write(df.info())

def clustering_page(df):
    st.title("Customer Data Exploration - Clustering")

    data = df.iloc[:, 3:5].values
    fig, ax = plt.subplots(figsize=(10, 7))
    ax.set_title("Customer Dendograms")
    dend = shc.dendrogram(shc.linkage(data, method='ward'))
    st.pyplot(fig)

    cluster = AgglomerativeClustering(n_clusters=5, linkage='ward')
    labels_ = cluster.fit_predict(data)

    fig, ax = plt.subplots(figsize=(10, 7))
    scatter = ax.scatter(data[:, 0], data[:, 1], c=cluster.labels_, cmap='rainbow')
    ax.set_title("Clusters")
    ax.set_xlabel("Feature 1")
    ax.set_ylabel("Feature 2")
    st.pyplot(fig)

def main():
    st.sidebar.title("Navigation")
    page = st.sidebar.selectbox("Choose a page", ["Overview", "Clustering"])

    df = load_data()

    if page == "Overview":
        overview_page(df)
    elif page == "Clustering":
        clustering_page(df)

if __name__ == "__main__":
    main()
