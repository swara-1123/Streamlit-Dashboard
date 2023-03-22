import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

st.markdown("<h1 style='text-align: center; color: black;'>Iris Flower Classification</h1>", unsafe_allow_html=True)
st.markdown("<h5 style='text-align: center; color: black;'>This is a web application for understanding the classification of iris flowers based on their different features.</h5>", unsafe_allow_html=True)
st.write('Iris flower classification is a very popular machine-learning project. The iris dataset contains three classes of flowers, Versicolor, Setosa, and Virginica, and each class contains 4 independent features, ‘Sepal length’, ‘Sepal width’, ‘Petal length’, and ‘Petal width’.  And Dependent feature, which will be the output for the model, is Species. It contains the name of the species to which that particular flower with those measurements belongs. The aim of the iris flower classification is to predict flowers based on their specific features.')
st.image("https://www.neuraldesigner.com/images/iris-flower.jpeg")

df = pd.read_csv(r"C:\Users\karni\OneDrive\Desktop\internship-innomatics\proj1\iris.csv")
st.header('Statistics of Dataframe')
st.write(df.describe())

st.header('Features of Iris Flower')
st.image("https://editor.analyticsvidhya.com/uploads/51518iris%20img1.png")

st.header('Head of Dataframe')
st.write(df.head())

st.header('Tail of the Dataframe')
st.write(df.tail())

st.header('Correlation Between Data')
st.header('i. Pairplot')
sns_plot=sns.pairplot(df, hue='Species',palette='Paired')
st.pyplot(sns_plot)

st.header('ii. Scatterplot')
fig, ax = plt.subplots(1,1)
ax.scatter(x=df['Species'], y=df['SepalLengthCm'])
ax.set_xlabel('Species')
ax.set_ylabel('SepalLengthCm')
st.pyplot(fig)

st.header('iii. Countplot')
fig = plt.figure(figsize=(10, 4))
sns.countplot(x = "Species", data = df, palette='Set2')
st.pyplot(fig)

st.header('iv. Heatmap')
fig = plt.figure(figsize=(10, 5))
sns.heatmap(df.corr(),cmap="flare", linecolor='white', linewidths=1,annot=True)
st.pyplot(fig)

st.header('v. Boxplot')
fig = plt.figure(figsize=(10, 5))
sns.boxplot(x='Species',y='SepalWidthCm',data=df,palette='rocket')
st.pyplot(fig)

fig = plt.figure(figsize=(10, 5))
sns.boxplot(x='Species',y='SepalLengthCm',data=df,palette='rocket')
st.pyplot(fig)

fig = plt.figure(figsize=(10, 5))
sns.boxplot(x='Species',y='PetalWidthCm',data=df,palette='rocket')
st.pyplot(fig)

fig = plt.figure(figsize=(10, 5))
sns.boxplot(x='Species',y='PetalLengthCm',data=df,palette='rocket')
st.pyplot(fig)