import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

@st.cache_data
def load_data():
    df = pd.read_csv("Fish.csv")
    return df

@st.cache_data
def train_model(df):
    X = df.drop("Species", axis=1)
    y = df["Species"]
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    accuracy = model.score(X_test, y_test)
    return model, accuracy

def main():
    st.set_page_config(page_title="Klasifikasi Spesies Ikan", layout="centered")
    st.title("Aplikasi Klasifikasi Spesies Ikan")
    st.write("Gunakan slider di samping untuk memasukkan data fisik ikan.")

    df = load_data()
    model, accuracy = train_model(df)
    st.sidebar.header("Parameter Fisik Ikan")
    
    weight = st.sidebar.slider("Weight (g)", float(df.Weight.min()), float(df.Weight.max()), float(df.Weight.mean()))
    length1 = st.sidebar.slider("Length1 (Vertical)", float(df.Length1.min()), float(df.Length1.max()), float(df.Length1.mean()))
    length2 = st.sidebar.slider("Length2 (Diagonal)", float(df.Length2.min()), float(df.Length2.max()), float(df.Length2.mean()))
    length3 = st.sidebar.slider("Length3 (Cross)", float(df.Length3.min()), float(df.Length3.max()), float(df.Length3.mean()))
    height = st.sidebar.slider("Height (cm)", float(df.Height.min()), float(df.Height.max()), float(df.Height.mean()))
    width = st.sidebar.slider("Width (cm)", float(df.Width.min()), float(df.Width.max()), float(df.Width.mean()))

    if st.button("Prediksi Spesies"):
        input_data = np.array([[weight, length1, length2, length3, height, width]])
        prediction = model.predict(input_data)[0]
        
        st.success(f"Ikan tersebut kemungkinan besar adalah spesies: **{prediction}**")
        st.info(f"Akurasi model: {accuracy*100:.2f}%")

    if st.checkbox("Tampilkan Tabel Data"):
        st.dataframe(df.head(10))

if __name__ == "__main__":
    main()