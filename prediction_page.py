import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from xgboost import XGBRegressor
import plotly.express as px

def show_prediction_page():
    # Load the data
    data = pd.read_csv(r"C:\Users\cengi\Desktop\streamlit\forecast.csv")

    # City selection
    selected_cities = st.multiselect(
        'Select cities',
        data['City'].unique(),
        default=data['City'].unique()
    )
    data = data[data['City'].isin(selected_cities)]

    # Column selection
    selected_columns = st.multiselect(
        'Select columns',
        data.columns,
        default=data.columns.tolist()
    )
    data = data[selected_columns]

    # Show the data
    if not data.empty:
        st.write("Selected Data:")
        st.dataframe(data)
    else:
        st.write("No data available for the selected cities and columns.")

    # Filter data for training
    selected_years_train = st.multiselect("Select years for training data", options=[2022, 2023], default=[2023])
    selected_months_train = st.multiselect("Select months for training data", options=range(1, 13), default=[2, 3])

    train_data = data[(data['Year'].isin(selected_years_train)) & (data['Month'].isin(selected_months_train))]

    # Filter data for testing
    selected_years_test = st.multiselect("Select years for testing data", options=[2024], default=[2024])
    selected_months_test = st.multiselect("Select months for testing data", options=range(1, 13), default=[2])

    test_data = data[(data['Year'].isin(selected_years_test)) & (data['Month'].isin(selected_months_test))]

    # Extract input features and target variable for training
    X_train_columns = st.multiselect("Select columns for X_train", train_data.columns, default=['142_area', '143_area', '143_rate'])
    y_train_column = st.selectbox("Select column for y_train", train_data.columns, index=train_data.columns.tolist().index('Reported_Wheat/Barley'))

    X_train = train_data[X_train_columns]
    y_train = train_data[y_train_column]

    # Model selection
    model_name = st.selectbox("Select model", options=["Random Forest", "Gradient Boosting", "XGBoost"])

    # Train the selected model
    if model_name == "Random Forest":
        model = RandomForestRegressor()
    elif model_name == "Gradient Boosting":
        model = GradientBoostingRegressor()
    elif model_name == "XGBoost":
        model = XGBRegressor()

    model.fit(X_train, y_train)

    # Predict for test data
    X_test = test_data[X_train_columns]
    predicted_wheat_barley = model.predict(X_test)

    # Create a DataFrame for predicted areas
    predicted_areas_df = pd.DataFrame({
        'Month': test_data['Month'],
        'City': test_data['City'],
        'District': test_data['District'],
        'Reported_Wheat/Barley': predicted_wheat_barley
    })

    # Show predicted wheat/barley production for selected months and year
    st.write("Predicted wheat/barley production:")
    st.dataframe(predicted_areas_df)

    # Generate Excel report
    if st.button("Generate Excel Report"):
        generate_excel_report(predicted_areas_df)

    # Get feature importances
    if model_name == "Random Forest" or model_name == "XGBoost":
        feature_importances = model.feature_importances_
    else:
        # For Gradient Boosting, we use SHAP or other methods for feature importances
        feature_importances = []

    # Create DataFrame to display feature importances
    feature_importance_df = pd.DataFrame({
        'Feature': X_train_columns,
        'Importance': feature_importances
    })

    # Sort features by importance (descending order)
    feature_importance_df = feature_importance_df.sort_values(by='Importance', ascending=False)

    # Show feature importances
    st.write("Feature Importances:")
    st.dataframe(feature_importance_df)

    # Plot feature importances as pie chart
    fig = px.pie(feature_importance_df, values='Importance', names='Feature', title='Feature Importance')
    st.plotly_chart(fig, use_container_width=True)

def generate_excel_report(predicted_areas_df):
    # Create Excel file
    with pd.ExcelWriter('predicted_areas.xlsx', engine='xlsxwriter') as writer:
        predicted_areas_df.to_excel(writer, index=False)
    
    # Download Excel file
    st.success("Excel report has been generated.")
    st.markdown("Download [Excel report](predicted_areas.xlsx)")

if __name__ == "__main__":
    show_prediction_page()
