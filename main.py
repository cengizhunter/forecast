import streamlit as st
import prediction_page
import plot_page
st.set_option('deprecation.showPyplotGlobalUse', False)


# Create a dictionary to map page names to functions
PAGES = {
    "Prediction": prediction_page.show_prediction_page,
    "Plots": plot_page.show_plot_page
}

# Add a sidebar to select the page to show
st.sidebar.title('Navigation')
selection = st.sidebar.radio("Go to", list(PAGES.keys()))

# Show the selected page
page = PAGES[selection]
page()
