import streamlit as st
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.neighbors import KNeighborsClassifier

@st.cache_data
def load_data():
    data = pd.read_csv('crop_production.csv')  
    data = data.dropna()
    
    # Encode categorical columns
    label_encoders = {}
    for column in ['State_Name', 'District_Name', 'Season', 'Crop']:
        le = LabelEncoder()
        data[column] = le.fit_transform(data[column])
        label_encoders[column] = le

    return data, label_encoders

data, label_encoders = load_data()

X = data[['State_Name', 'District_Name', 'Season']]
y = data['Crop']

@st.cache_resource
def train_knn(X, y):
    knn = KNeighborsClassifier(n_neighbors=10)  # Increase neighbors to get more recommendations and avoid duplicates
    knn.fit(X, y)
    return knn

knn_model = train_knn(X, y)

st.title("Crop Recommendation System")
state_name = st.selectbox("Select State", label_encoders['State_Name'].classes_)
state_encoded = label_encoders['State_Name'].transform([state_name])[0]
filtered_districts = data[data['State_Name'] == state_encoded]
unique_districts = label_encoders['District_Name'].inverse_transform(filtered_districts['District_Name'].unique())

district_name = st.selectbox("Select District", unique_districts)
season = st.selectbox("Select Season", label_encoders['Season'].classes_)
district_encoded = label_encoders['District_Name'].transform([district_name])[0]
season_encoded = label_encoders['Season'].transform([season])[0]
user_input = [[state_encoded, district_encoded, season_encoded]]

# Get nearest crops and filter to unique crop names
distances, indices = knn_model.kneighbors(user_input)

# Convert indices to unique crop names
top_crops = []
seen_crops = set()
for idx in indices[0]:
    crop = y.iloc[idx]
    if crop not in seen_crops:
        seen_crops.add(crop)
        crop_name = label_encoders['Crop'].inverse_transform([crop])[0]
        top_crops.append(crop_name)

# Display unique crop recommendations
st.write(f"### Top Recommended Crops for {season} in {district_name}, {state_name}:")
for rank, crop_name in enumerate(top_crops, start=1):
    st.write(f"{rank}. **{crop_name}**")
