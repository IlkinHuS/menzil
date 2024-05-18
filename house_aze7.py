import streamlit as st
import pandas as pd
import numpy as np
import xgboost as xgb
import joblib 
import geohash2 as gh2
import plotly.graph_objects as go
import folium
from streamlit_folium import st_folium

# Load the trained model
binamodel = joblib.load('C:/Users/ssg-user/streamlit/binaaz1500.pkl')

data_structure = pd.read_csv('C:/Users/ssg-user/streamlit/data_format2.csv')

# Function to preprocess input data
def preprocess_input(data):
    tr2 = pd.DataFrame(columns=data_structure.columns)
    tr2.loc[0] = 0
    tr2.loc[0, 'otaq_sayi'] = data['otaq_sayi'][0]
    tr2.loc[0, 'sahe_kvm'] = np.log(data['sahe_kvm'][0])
    tr2.loc[0, 'mertebe_yer'] = data['mertebe_yer'][0]
    tr2.loc[0, 'mertebe_say'] = data['mertebe_say'][0]
    tr2.loc[0, 'kateqoriyaYeni tikili'] = 1 if data['kateqoriya'][0] == 'Yeni tikili' else 0
    tr2.loc[0, 'ipotekanan'] = 1 if data['ipoteka'][0] != 'var' else 0
    tr2.loc[0, 'temiryoxdur'] = 1 if data['temir'][0] != 'var' else 0
    tr2.loc[0, 'geohash' + data['geohash'][0]] = 1
    if data['year_month'][0] != '2023-09':
        tr2.loc[0, 'year_month' + data['year_month'][0]] = 1
    return tr2

# Streamlit app
def main():
    menu = ["Visual", "Mənzil qiymətləndirməsi", "Kirayə qiymətləndirməsi", "..."]
    choices = st.sidebar.selectbox("Menu", menu)

    if choices == 'Mənzil qiymətləndirməsi':
        st.title("Mənzil Qiymətini Müəyyənləşdir")

        # Input fields
        otaq_sayi = st.number_input("Otaq sayı", min_value=1, value=4, max_value=15)
        sahe_kvm = st.number_input("Sahə (kv.m)", min_value=10, value=90, max_value=500)
        mertebe_yer = st.number_input("Mərtəbə", min_value=0, value=5, max_value=50)
        mertebe_say = st.number_input("Mərtəbə sayı", min_value=1, value=9, max_value=50)
        kateqoriya = st.selectbox("Kateqoriya", ['Yeni tikili', 'Köhnə tikili'])
        col1, col2 = st.columns([1, 1])
        with col1:
            ipoteka = st.radio("İpoteka", ['var', 'yoxdur'])
        with col2:
            temir = st.radio("Təmir", ['var', 'yoxdur'])

        # Map for selecting location
        st.write("Select the location on the map:")
        default_location = [40.400222, 49.7956866]

        # Initialize folium map
        m = folium.Map(location=default_location, zoom_start=12)

        # Add a marker to the map that updates on click
        marker = folium.Marker(location=default_location, draggable=True)
        marker.add_to(m)

        # Display the map
        map_data = st_folium(m, width=700, height=500)

        # Update marker position based on user click
        if map_data and map_data['last_clicked']:
            latitude = map_data['last_clicked']['lat']
            longitude = map_data['last_clicked']['lng']
            # Update the marker position
            marker.location = [latitude, longitude]
            folium.Marker(location=[latitude, longitude], draggable=True).add_to(m)
            st.write(f"Selected location: Latitude: {latitude}, Longitude: {longitude}")
        else:
            latitude = default_location[0]
            longitude = default_location[1]

        year_month = st.selectbox("Qimətləndirmə tarixi", ['2024-04', '2024-03', '2024-02', '2024-01', '2023-12', '2023-11', '2023-10'])

        # Predict button
        data = {
            'otaq_sayi': [otaq_sayi],
            'sahe_kvm': [sahe_kvm],
            'mertebe_yer': [mertebe_yer],
            'mertebe_say': [mertebe_say],
            'kateqoriya': [kateqoriya],
            'ipoteka': [ipoteka],
            'temir': [temir],
            'geohash': [gh2.encode(latitude, longitude, precision=6)],
            'year_month': [year_month]
        }

        if st.button("Qiymətləndir"):
            tr2 = preprocess_input(data)
            predicted_price = binamodel.predict(tr2)[0]
            predicted_price = np.exp(predicted_price)
            st.success(f"Predicted house price: {predicted_price:.2f} AZN")

        show_graph = st.checkbox("Digər dövrlər üzrə qiymətləndir")
        # Plot predicted prices for all possible values of year_month
        if show_graph:
            st.subheader("")
            year_months = ['2023-10', '2023-11', '2023-12', '2024-01', '2024-02', '2024-03', '2024-04']
            predicted_prices = []
            for month in year_months:
                data['year_month'] = [month]
                tr2 = preprocess_input(data)
                predicted_price = binamodel.predict(tr2)[0]
                predicted_prices.append(np.exp(predicted_price))

            fig = go.Figure(data=go.Scatter(x=year_months, y=predicted_prices, mode='lines+markers'))
            fig.update_layout(title='Dövrlər üzrə mənzilin qiyməti',
                              xaxis_title='Tarix',
                              yaxis_title='Qiymət (AZN)')
            st.plotly_chart(fig)

    if choices == 'Visual':
        st.title('Interactive Map')
        # Load and display the HTML file containing the figure
        st.components.v1.html(open('C:/Users/ssg-user/streamlit/hyperlink_fig34.html').read(), width=800, height=600)

if __name__ == "__main__":
    main()
