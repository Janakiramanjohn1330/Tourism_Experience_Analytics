import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from lightgbm import LGBMClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import numpy as np


st.set_page_config(page_title="Tourist Recommendation App", layout="wide")
st.title("🌍 Tourist Attraction Recommender")
st.sidebar.header("INFO")

side_bar = st.sidebar.selectbox('', options=['INTRODUCTION', 'CLASSIFICATION', 'REGRESSION', 'EDA'])


if side_bar== 'INTRODUCTION':
        st.markdown("""
    Welcome to the **Tourist Attraction Recommender App**! 🌍  
    This interactive tool helps you explore, analyze, and predict tourist preferences across different regions.  
    Use the **Classification** tab to predict how users might visit attractions, or try **Regression** to forecast ratings.  
    Dive into the **EDA** section for visual insights on destinations, ratings, and travel trends.  
    Perfect for travel analysts, tourism boards, or curious explorers! ✈️🗺️
    """)


# -------------------- CLASSIFICATION SECTION --------------------
if side_bar == 'CLASSIFICATION':
    st.title('🧠 Classification Model Prediction')

    clf_model = st.selectbox('Choose a classifier model:',
                             options=['Random Forest', 'Logistic Regression', 'XGBoost', 'LightGBM'])

    # Load and preprocess data
    data = pd.read_csv("c:/Users/Admin/Desktop/DS_PROJECT/Tourism_ML_project/fillna dataset.csv")
    categorical_data = ['CityName', 'Country', 'Region', 'Continent', 'VisitMode', 'Attraction', 'AttractionType']
    label_encoders = {}

    for col in categorical_data:
        le = LabelEncoder()
        data[col] = le.fit_transform(data[col])
        label_encoders[col] = le

    X = data.drop(columns=['VisitMode'])
    y = data['VisitMode']
    x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Choose classifier
    if clf_model == 'Random Forest':
        model = RandomForestClassifier(n_estimators=100, random_state=42)
    elif clf_model == 'Logistic Regression':
        model = LogisticRegression(max_iter=1000)
    elif clf_model == 'XGBoost':
        model = XGBClassifier(estimators=100,learning_rate=0.1, eval_metric='mlogloss')
    elif clf_model == 'LightGBM':
        model = LGBMClassifier()

    model.fit(x_train, y_train)
    y_pred = model.predict(x_test)

    acc = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, average='weighted')
    recall = recall_score(y_test, y_pred, average='weighted')
    report = classification_report(y_test, y_pred)

    st.subheader("🔍 Evaluation Metrics")
    st.write(f"**Accuracy:** {acc:.2f}")
    st.write(f"**F1 Score:** {f1:.2f}")
    st.write(f"**Recall:** {recall:.2f}")
    st.text("Classification Report:")
    st.text(report)

# -------------------- REGRESSION SECTION --------------------
if side_bar == 'REGRESSION':
    st.title('📈 Regression Model Prediction')

    reg_model = st.selectbox('Choose a Regression model:',
                             options=['Linear Regression', 'Random Forest Regression', 'Decision Tree', 'Gradient Boosting'])

    # Load and preprocess data
    data = pd.read_csv("c:/Users/Admin/Desktop/DS_PROJECT/Tourism_ML_project/fillna dataset.csv")
    categorical_data = ['CityName', 'Country', 'Region', 'Continent', 'VisitMode', 'Attraction', 'AttractionType']
    label_encoders = {}

    for col in categorical_data:
        le = LabelEncoder()
        data[col] = le.fit_transform(data[col])
        label_encoders[col] = le

    X = data.drop(columns=['Rating'])
    y = data['Rating']
    x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Choose regression model
    if reg_model == 'Random Forest Regression':
        model = RandomForestRegressor(n_estimators=100, random_state=42)
    elif reg_model == 'Linear Regression':
        model = LinearRegression()
    elif reg_model == 'Decision Tree':
        model = DecisionTreeRegressor(random_state=42)
    elif reg_model == 'Gradient Boosting':
        model = GradientBoostingRegressor(n_estimators=100, random_state=42)

    model.fit(x_train, y_train)
    y_pred = model.predict(x_test)

    mse = mean_squared_error(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    st.subheader("🔍 Evaluation Metrics")
    st.write(f"**Mean Squared Error (MSE):** {mse:.2f}")
    st.write(f"**Mean Absolute Error (MAE):** {mae:.2f}")
    st.write(f"**R² Score:** {r2:.2f}")


if side_bar == 'EDA':
    st.title("📊 Exploratory Data Analysis")

    eda = pd.read_csv("c:/Users/Admin/Desktop/DS_PROJECT/Tourism_ML_project/fillna dataset.csv")

    charts = st.selectbox('Select a visualization:',
                          options=['User Distribution Across Countries',
                                   'Distribution Across Continents',
                                   'User Distribution Across Region',
                                   'Attraction Types Based on Rating',
                                   'Average Ratings Across Regions',
                                   'Average Ratings Across Attractions'])

    if charts == 'User Distribution Across Countries':
        fig, ax = plt.subplots(figsize=(12, 5))
        sns.countplot(y=eda['Country'], order=eda['Country'].value_counts().nlargest(10).index, palette='viridis', ax=ax)
        ax.set_title('User Distribution Across Countries')
        ax.set_xlabel('Number of Users')
        ax.set_ylabel('Country')
        st.pyplot(fig)

    elif charts == 'Distribution Across Continents':
        fig, ax = plt.subplots(figsize=(8, 6))
        sns.countplot(x=eda['Continent'], palette='coolwarm', ax=ax)
        ax.set_title('Distribution Across Continents')
        ax.set_ylabel('Number of Users')
        st.pyplot(fig)

    elif charts == 'User Distribution Across Region':
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.countplot(y=eda['Region'], order=eda['Region'].value_counts().nlargest(10).index, palette='Set2', ax=ax)
        ax.set_title('User Distribution Across Regions')
        ax.set_xlabel('Number of Users')
        st.pyplot(fig)

    elif charts == 'Attraction Types Based on Rating':
        top_attraction_types = eda.groupby('AttractionType')['Rating'].mean().sort_values(ascending=False).head(10)

        fig, ax = plt.subplots(figsize=(10, 6))
        sns.barplot(x=top_attraction_types.values, y=top_attraction_types.index, palette='pastel', ax=ax)
        ax.set_title('Top 10 Attraction Types by Average Rating')
        ax.set_xlabel('Average Rating')
        ax.set_ylabel('Attraction Type')
        st.pyplot(fig)


    elif charts == 'Average Ratings Across Regions':
        avg_rating_region = eda.groupby('Region')['Rating'].mean().sort_values(ascending=False)
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.barplot(x=avg_rating_region.values, y=avg_rating_region.index, palette='Blues_r', ax=ax)
        ax.set_title('Average Ratings Across Regions')
        ax.set_xlabel('Average Rating')
        ax.set_ylabel('Region')
        st.pyplot(fig)

    elif charts == 'Average Ratings Across Attractions':
        avg_rating_attractions = eda.groupby('Attraction')['Rating'].mean().sort_values(ascending=False).head(10)
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.barplot(x=avg_rating_attractions.values, y=avg_rating_attractions.index, palette='magma', ax=ax)
        ax.set_title('Top 10 Attractions by Average Rating')
        ax.set_xlabel('Average Rating')
        ax.set_ylabel('Attraction')
        st.pyplot(fig)
