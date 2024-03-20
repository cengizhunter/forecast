import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
st.set_option('deprecation.showPyplotGlobalUse', False)
def show_plot_page():
    # Load the data
    data = pd.read_csv("forecast.csv")

    # City selection
    selected_cities = st.multiselect(
        'Select cities',
        data['City'].unique(),
        default=data['City'].unique()
    )

    # Filter data for selected cities
    selected_data = data[data['City'].isin(selected_cities)]

    # Calculate average 143_area value for each month for each city
    average_143_area = selected_data.groupby(['City', 'Month'])['143_area'].mean().reset_index()

    # Plot time series for each city
    plt.figure(figsize=(12, 6))
    for city in selected_cities:
        city_data = average_143_area[average_143_area['City'] == city]
        sns.lineplot(x='Month', y='143_area', data=city_data, label=city)

    plt.title('Average 143 Area by Month for Selected Cities')
    plt.xlabel('Month')
    plt.ylabel('Average 143 Area')
    plt.legend()
    st.pyplot()
