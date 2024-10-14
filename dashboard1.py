import streamlit as st
import seaborn as sns
import plotly.express as px
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import time
from scipy.stats import gaussian_kde

# Set the page title
st.set_page_config(page_title="Titanic Dashboard", layout="wide")

# Load the Titanic dataset
@st.cache
def load_titanic_data():
    return sns.load_dataset("titanic")

titanic = load_titanic_data()

# Sidebar menu for navigation
page = st.sidebar.selectbox("Navigation", ["Overview", "Seaborn Charts", "Plotly Charts", "Additional Insights and Predictions", "About"])

# Title of the page
st.title(f"Titanic Data - {page}")

### 1. Overview Page ###
if page == "Overview":
    st.header("Dataset Overview")
    
    # Show the first few rows of the dataset
    st.write("### First Five Rows of the Dataset")
    st.dataframe(titanic.head())

    # Show some key metrics
    st.write("### Key Metrics")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Total Passengers", titanic.shape[0])
    with col2:
        st.metric("Survived", f"{round(titanic['survived'].mean() * 100, 2)}%")
    with col3:
        st.metric("Average Age", f"{round(titanic['age'].mean(), 2)} years")
    
    st.write("### Dataset Summary")
    st.write(titanic.describe())

    # Simulate progress bar for data loading
    st.subheader("Simulating Data Processing")
    progress_bar = st.progress(0)
    for percent_complete in range(100):
        time.sleep(0.01)
        progress_bar.progress(percent_complete + 1)

### 2. Seaborn Charts Page ###
if page == "Seaborn Charts":
    st.header("Seaborn Charts")
    
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Histogram: Age Distribution")
        fig, ax = plt.subplots()
        sns.histplot(data=titanic, x="age", bins=20, kde=True, ax=ax, color="purple")
        st.pyplot(fig)
    
    with col2:
        st.subheader("Scatterplot: Age vs Fare")
        fig, ax = plt.subplots()
        sns.scatterplot(data=titanic, x="age", y="fare", hue="survived", style="sex", ax=ax, palette="coolwarm")
        st.pyplot(fig)

    st.write("### Correlation Heatmap")
    corr_matrix = titanic.select_dtypes(include='number').corr()
    fig, ax = plt.subplots()
    sns.heatmap(corr_matrix, annot=True, cmap='Purples', ax=ax)
    st.pyplot(fig)

    # KDE Plot for Fare
    st.subheader("KDE Plot: Fare Distribution")
    fig, ax = plt.subplots()
    sns.kdeplot(titanic["fare"].dropna(), bw_adjust=0.5, fill=True, color="magenta", ax=ax)
    ax.set_title("Probability Density Function of Titanic Fares")
    st.pyplot(fig)

### 3. Plotly Charts Page ###
if page == "Plotly Charts":
    st.header("Plotly Charts")
    
    # Display a scatterplot and a bar chart side by side using columns
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Scatterplot: Age vs Fare")
        scatter_fig = px.scatter(titanic, x="age", y="fare", color="sex", symbol="survived", 
                                 title="Fare vs Age", color_discrete_sequence=["purple", "blue"])
        st.plotly_chart(scatter_fig)
    
    with col2:
        st.subheader("Bar Chart: Class vs Fare")
        custom_colors = ['#8A2BE2', '#FF69B4', '#1E90FF']  # Purple, Pink, Blue
        bar_fig = px.bar(titanic, x="class", y="fare", color="class", barmode="group", 
                         title="Class vs Fare", color_discrete_sequence=custom_colors)
        st.plotly_chart(bar_fig)

    # Interaction Plot: Passenger Class and Sex on Fare
    st.subheader("Interaction Between Passenger Class and Sex on Fare")
    fig, ax = plt.subplots()
    sns.pointplot(x="pclass", y="fare", hue="sex", data=titanic, markers=["o", "x"], linestyles=["-", "--"], palette="coolwarm", ax=ax)
    ax.set_title("Interaction Between the Passenger Class and Sex on Fare")
    st.pyplot(fig)

### 4. Survival Analysis Page ###
if page == "Survival Analysis":
    st.header("Survival Analysis")
    
    # Survival Count by Gender
    st.subheader("Survival Count by Gender")
    survival_by_gender = titanic.groupby('sex')['survived'].value_counts(normalize=True).unstack() * 100
    fig, ax = plt.subplots()
    survival_by_gender.plot(kind='bar', stacked=True, ax=ax, colormap="Purples")
    ax.set_ylabel("Survival Rate (%)")
    st.pyplot(fig)

    # Survival Rate by Age Group
    st.subheader("Survival Rate by Age Group")
    titanic['age_group'] = pd.cut(titanic['age'], bins=[0, 18, 35, 60, 80], labels=["Child", "Young Adult", "Adult", "Senior"])
    survival_by_age_group = titanic.groupby('age_group')['survived'].mean() * 100
    st.bar_chart(survival_by_age_group)

    # Actual vs Predicted Fare Plot
    st.subheader("Actual vs Predicted Fare")
    # Simulated y and predictions for illustration purposes
    y = titanic['fare'].dropna()
    predictions = y * np.random.uniform(0.9, 1.1, size=y.shape[0])  # Simulate predicted values
    r_squared = 0.85  # Simulate R2 value
    fig, ax = plt.subplots()
    ax.scatter(y, y, color='blue', label='Actual Values')
    ax.scatter(y, predictions, color='red', label='Predicted Values')
    ax.plot([min(y), max(y)], [min(y), max(y)], color='black', linestyle='--', label="Perfect Fit")
    ax.set_xlabel('Actual Fare')
    ax.set_ylabel('Predicted Fare')
    ax.set_title(f'Actual vs. Predicted Fare (R2 = {r_squared})')
    ax.legend()
    st.pyplot(fig)

### 5. About Page ###
if page == "About":
    st.header("About this Dashboard")
    st.write("""
        This dashboard provides insights into the Titanic dataset using Seaborn, Plotly, and custom visualizations.
        The graphs are designed with a focus on presenting clear and insightful data visualizations in shades of purple, pink, and blue.
    """)
