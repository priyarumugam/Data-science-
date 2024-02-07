import streamlit as st
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from mlxtend.preprocessing import TransactionEncoder
from mlxtend.frequent_patterns import apriori, association_rules
import plotly.express as px  

dataset_path = "C:\\Users\\Priya\\Desktop\\projects on algorithms\\apriori\\Market_Basket_Optimisation.csv"

df = pd.read_csv(dataset_path)

transaction = []
for i in range(0, df.shape[0]):
    for j in range(0, df.shape[1]):
        transaction.append(df.values[i, j])

transaction = np.array(transaction)
df = pd.DataFrame(transaction, columns=["items"])
df["incident_count"] = 1
indexNames = df[df['items'] == "nan"].index
df.drop(indexNames, inplace=True)
df_table = df.groupby("items").sum().sort_values("incident_count", ascending=False).reset_index()

df_table["all"] = "all"
fig = px.treemap(df_table.head(30), path=['all', "items"], values='incident_count',
                  color=df_table["incident_count"].head(30), hover_data=['items'],
                  color_continuous_scale='Greens',
                )

te = TransactionEncoder()
te_ary = te.fit(transaction).transform(transaction)
dataset = pd.DataFrame(te_ary, columns=te.columns_)

frequent_itemsets = apriori(dataset, min_support=0.01, use_colnames=True)
frequent_itemsets['length'] = frequent_itemsets['itemsets'].apply(lambda x: len(x))

rules = association_rules(frequent_itemsets, metric="lift", min_threshold=1.2)
rules["antecedents_length"] = rules["antecedents"].apply(lambda x: len(x))
rules["consequents_length"] = rules["consequents"].apply(lambda x: len(x))
rules_sorted_lift = rules.sort_values("lift", ascending=False)
rules_sorted_confidence = rules.sort_values("confidence", ascending=False)

st.title("Market Basket Analysis Web App")

st.subheader("Treemap Visualization")
st.plotly_chart(fig)

st.subheader("Association Rules")
st.dataframe(rules_sorted_lift.head(10))
