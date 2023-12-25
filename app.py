import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
import pickle
import time 
# Load the model
RFR_model = pickle.load(open('your_model_filename.pkl', 'rb'))
# print(type(RFR_model))

with open('label_encoders.pkl', 'rb') as f:
    label_encoders = pickle.load(f)
# print('lblencooooooood',label_encoders.keys())

with open('scaler.pkl', 'rb') as f:
    scaler = pickle.load(f)

# Define feature labels (no need to specify encodings manually)
feature_labels = {
    'assembly': {'Imported', 'Local'},
    'body': {'Hatchback', 'Station Wagon', 'Off-Road Vehicles', 'Van', 'Single Cabin', 'Pick Up', 'Mini Vehicles', 'Compact sedan', 'Convertible', 'Compact SUV', 'High Roof', 'Crossover', 'Sedan', 'MPV', 'Truck', 'Mini Van', 'Coupe', 'Double Cabin', 'SUV', 'Micro Van'},
    'make': {'Changan', 'United', 'Daewoo', 'Subaru', 'Land', 'Volkswagen', 'Ford', 'Range', 'Proton', 'Prince', 'KIA', 'MG', 'Honda', 'Daihatsu', 'Hyundai', 'Porsche', 'Isuzu', 'Haval', 'BMW', 'Mazda', 'BAIC', 'Lexus', 'Chevrolet', 'Jeep', 'Mercedes', 'FAW', 'Mitsubishi', 'Audi', 'Chery', 'Nissan', 'Suzuki', 'Toyota', 'DFSK'},
    'model': {'Sonata', 'Rav4', 'Pride', 'Succeed', 'Carry', 'Mark', 'Carrier', 'Clipper', 'Lancer', 'LX', 'Alpha', 'Taft', 'Vitz', 'Stream', 'Karvaan', 'Glory', 'C-HR', 'Cx3', 'Mirage', 'Excel', 'Voxy', 'Potohar', 'Noah', 'Fortuner', 'Bravo', 'I', 'Belta', 'Freed', 'Prius', 'Ek', 'Tiida', 'Saga', 'Avanza', 'CR-Z', 'Insight', 'K07', 'HS', 'N', 'Accord', 'M', 'Ciaz', 'Surf', 'Mira', 'Moco', 'Kizashi', 'Tanto', 'A5', 'Bluebird', 'Pixis', 'Corolla', 'March', 'Shehzore', 'Yaris', 'CT200h', 'Kei', 'Wingroad', 'Life', 'X1', 'RX', 'Liana', 'Vezel', 'Rush', 'Optra', 'Tank', 'Corona', 'Vamos', 'H6', 'Serena', 'Mehran', 'Roomy', 'X70', 'Passo', 'X5', 'Fj', 'X-PV', 'Civic', 'FX', 'Margalla', 'Harrier', 'Wrangler', 'RX8', 'Allion', 'Bolan', 'MR', 'Grace', 'IST', 'Xbee', 'HR-V', 'Fit', 'Hiace', 'Pajero', 'Charade', 'Platz', 'Khyber', 'Otti', 'AD', 'Picanto', 'Carol', 'Crown', 'Beetle', 'Sunny', 'Wish', 'Hijet', 'Hustler', 'A3', 'Every', 'Mega', 'Swift', 'ZS', 'Sirius', 'Q7', 'Sorento', 'CR-V', 'Elantra', 'H-100', 'Raize', 'Cross', 'Land', 'Tucson', 'Joy', 'Probox', 'Prado', 'Rocky', 'Spectra', 'M9', 'Rover', 'Classic', 'Q3', 'Tiggo', 'Cast', 'Duet', 'Alto', 'Ravi', 'Spacia', 'Grand', 'Esse', 'Juke', 'Move', 'City', 'Racer', 'Dayz', 'Flair', 'A6', 'Wagon', 'Estima', 'Zest', 'Sportage', 'Cultus', 'Hilux', 'Sienta', 'Note', 'Alsvin', 'Copen', 'Oshan', 'Starlet', 'Terios', 'Cuore', 'BR-V', 'Aygo', 'ISIS', 'Aqua', 'Jolion', 'Jade', 'CJ', 'QQ', 'Airwave', 'Baleno', 'Pearl', 'Boon', 'Rvr', 'Galant', 'Exclusive', 'Stonic', 'D-Max', 'Cayenne', 'Roox', 'A4', 'Camry', 'iQ', 'Spike', 'Santro', 'Pleo', 'Jimny', 'BJ40', 'EK', 'Patrol', 'Vitara', 'Stella', 'Pino', 'F', 'APV', 'Acty', 'Benz', 'Premio', 'V2'},
    'transmission':  {'Automatic', 'Manual'},
    'color': {'silver', 'graphite', 'burgundy', 'pink', 'orange', 'brown', 'white', 'steel', 'navy', 'green', 'turquoise', 'titanium', 'indigo', 'red', 'blue', 'purple', 'yellow', 'bronze', 'gray', 'maroon', 'magenta', 'gold', 'black', 'beige'},
    'registered': {'Jhang', 'Gujranwala', 'Bannu', 'Mardan', 'Sahiwal', 'Peshawar', 'Kohat', 'Chakwal', 'Jhelum', 'Swat', 'Rahim Yar Khan', 'Attock', 'Okara', 'Rawalpindi', 'Kasur', 'Sukkur', 'Karachi', 'Vehari', 'Abbottabad', 'Bahawalnagar', 'Haripur', 'Mian Wali', 'Rawalakot', 'Sargodha', 'Muzaffarabad', 'Islamabad', 'Punjab', 'Sindh', 'D.G.Khan', 'Khanewal', 'Un-Registered', 'Hyderabad', 'Gujrat', 'Mirpur A.K.', 'Sialkot', 'Kashmir', 'Nowshera', 'Mansehra', 'Gilgit', 'Bhakkar', 'Lahore', 'Bahawalpur', 'Multan', 'Dera ismail khan', 'Quetta', 'Faisalabad', 'Sheikhupura'},
    'year': None,
    'mileage': None,
    'engine': None
}

# Define feature transformation function (with automatic encoding)
def transform_features(data):
    
    for feature in label_encoders.keys():
        if feature in data.columns:  # Ensure feature exists in new data
            data[feature] = label_encoders[feature].transform(data[feature])

    # print('bef:',data)
    # Apply scaling to all features
    data = scaler.transform(data)
    # print('aft:',data)
    # Reshape data into input vector format
    input_data = np.array([[data[0, 0], data[0, 1], data[0, 2], data[0, 3], data[0, 4],
                             data[0, 5], data[0, 6], data[0, 7], data[0, 8], data[0, 9]]],
                           dtype=object)
    return input_data

# Display an image from a local file
image_path = "car_pic.jpeg"
st.image(image_path, use_column_width=True)


# Create the Streamlit app (rest of the code remains the same)
st.title("Car Price Prediction")

# Get user input for features
make = st.selectbox("Enter Car Company", options=feature_labels['make'])
model = st.selectbox("Enter Car Model", options=feature_labels['model'])
body = st.selectbox("Enter Body Type", options=feature_labels['body'])
color = st.selectbox("Enter Color", options=feature_labels['color'])
assembly = st.selectbox("Enter Assembly Type", options=feature_labels['assembly'])
year = st.number_input("Enter Year of Production", min_value=1990, max_value=2022)
engine = st.number_input("Enter Engine Capacity (cc)", min_value=660, max_value=4000)
transmission = st.selectbox("Enter Transmission Type", options=feature_labels['transmission'])
registered = st.selectbox("Enter City of Registeration", options=feature_labels['registered'])
mileage = st.number_input("Enter Mileage", min_value=0)


# Collect input data into a DataFrame
data = pd.DataFrame({
    'assembly': [assembly],
    'body': [body],
    'make': [make],
    'model': [model],
    'year': [year],
    'engine': [engine],
    'transmission': [transmission],
    'color': [color],
    'registered': [registered],
    'mileage': [mileage]
})
# print('data:',data)
# Transform features
input_data = transform_features(data)
# print('data:',data)
# print('scaled data:',input_data)
# Predict price
# Predict price
if st.button("Predict Car Price"):
    with st.spinner("Predicting..."):
        # Simulate a delay (replace this with your actual prediction logic)
        time.sleep(1)
        
        prediction = RFR_model.predict(input_data)
        rounded_prediction = round(prediction[0])  # Round to zero decimals

        # Format the rounded prediction with commas
        formatted_prediction = f"{rounded_prediction:,.0f}"

        # Display the formatted prediction in bold
        st.success(f"**Predicted price: Rs. {formatted_prediction}**")

