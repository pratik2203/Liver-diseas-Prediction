import streamlit as st
import joblib
import numpy as np
import pandas as pd

# Load the trained model
model = joblib.load('model.pkl')
scaler = joblib.load('scaler.pkl')
# Define the Streamlit app
st.title("Liver Disease Prediction App")

# Sidebar for input features
st.sidebar.header("Input Features")

# Dictionary for categories, descriptions, precautions, and local image paths
category_info = {
    0: ("No Disease",
        "Congratulations! Your liver health looks fantastic! Keep up the great work, and remember that a balanced diet and healthy lifestyle are key to maintaining it. "
        "Continue eating nutritious foods, staying hydrated, and making liver-friendly choices to support your well-being. Here’s to your continued health and vitality!",
        "Keep up a balanced diet rich in fruits, vegetables, and whole grains. Avoid alcohol and processed foods as much as possible. Regular exercise and hydration are beneficial for liver health.",
        "https://media.istockphoto.com/id/1348797525/vector/best-foods-for-happy-liver-infographic-poster-cute-intestine-organ-character-vector-cartoon.jpg?s=170667a&w=0&k=20&c=GBwrzW3zEvAMgEITyN67OpeLT1r4y6c0-z1V1fmYe8s="),  # Update with the correct path on your system
    1: ("Cirrhosis",
        "Cirrhosis is advanced liver scarring due to prolonged liver damage, often from long-term alcohol use, viral hepatitis, or fatty liver disease. It can lead to liver failure if untreated.",
        "Avoid alcohol completely. Follow a liver-friendly diet that’s low in sodium and rich in vitamins and minerals. Regularly monitor liver function with your healthcare provider, and consider medication or treatment for underlying causes if advised.",
        "https://lirp.cdn-website.com/69c0b277/dms3rep/multi/opt/what+is+cirrhosis+of+the+liver-640w.jpg"),
    2: ("Hepatitis",
        "Hepatitis is liver inflammation, commonly caused by viral infections (like hepatitis A, B, or C) or autoimmune conditions. Chronic hepatitis can lead to scarring and liver damage if not managed.",
        "Get vaccinated for hepatitis A and B if at risk. Practice good hygiene, avoid sharing personal items, eat a nutritious low-fat diet, avoid alcohol, and follow prescribed treatments for viral hepatitis if diagnosed. Regular check-ups are essential to monitor liver health.",
        "https://lirp.cdn-website.com/69c0b277/dms3rep/multi/opt/Hepatitis+-+PACE+hospitals-640w.jpg"),
    3: ("Fibrosis",
        "Fibrosis is the initial scarring stage in the liver caused by chronic damage. Although milder than cirrhosis, it can progress if the underlying cause is not managed.",
        "Reduce alcohol intake and avoid hepatotoxic substances. Eat a liver-friendly diet, exercise regularly, and manage any conditions that might contribute to liver stress, such as diabetes or obesity. Regular liver check-ups can help track and prevent progression.",
        "https://st4.depositphotos.com/1042575/21615/v/450/depositphotos_216152866-stock-illustration-liver-fibrosis-concept-vector-illustration.jpg"),
    4: ("Suspect Disease",
        "This category indicates potential signs of liver dysfunction, though not clearly diagnosable as a specific condition. It’s a cautionary sign to investigate further.",
        "Consult a healthcare provider for a thorough assessment and follow recommended lifestyle adjustments. Avoid alcohol, eat a balanced diet, and stay active. Timely medical intervention can prevent further liver issues if any early signs of disease are confirmed.",
        "https://lirp.cdn-website.com/69c0b277/dms3rep/multi/opt/Prevention+of+End+stage+liver+disease+ESLD-640w.jpg"),
}

# Define input features in the sidebar
age = st.sidebar.number_input("Age")
sex = st.sidebar.selectbox("Sex", options=[0, 1], format_func=lambda x: "Male" if x == 1 else "Female")
albumin = st.sidebar.number_input("Albumin (14.9 - 82.2)", min_value=14.9, max_value=82.2, value=40.0)
alkaline_phosphatase = st.sidebar.number_input("Alkaline Phosphatase (11.3 - 416.6)", min_value=11.3, max_value=416.6, value=100.0)
alanine_aminotransferase = st.sidebar.number_input("Alanine Aminotransferase (0.9 - 325.3)", min_value=0.9, max_value=325.3, value=40.0)
aspartate_aminotransferase = st.sidebar.number_input("Aspartate Aminotransferase (10.6 - 324.0)", min_value=10.6, max_value=324.0, value=30.0)
bilirubin = st.sidebar.number_input("Bilirubin (0.8 - 209.0)", min_value=0.8, max_value=209.0, value=1.0)
cholinesterase = st.sidebar.number_input("Cholinesterase (1.42 - 16.41)", min_value=1.42, max_value=16.41, value=10.0)
cholesterol = st.sidebar.number_input("Cholesterol (1.43 - 9.67)", min_value=1.43, max_value=9.67, value=5.0)
creatinina = st.sidebar.number_input("Creatinine (8.0 - 1079.1)", min_value=8.0, max_value=1079.1, value=80.0)
gamma_glutamyl_transferase = st.sidebar.number_input("Gamma-Glutamyl Transferase (4.5 - 650.9)", min_value=4.5, max_value=650.9, value=50.0)
protein = st.sidebar.number_input("Protein (0 - 90.0)", max_value=90.0, value=70.0)

# Prepare the feature array for prediction
features = np.array([[age, sex, albumin, alkaline_phosphatase, alanine_aminotransferase,
                      aspartate_aminotransferase, bilirubin, cholinesterase, cholesterol,
                      creatinina, gamma_glutamyl_transferase, protein]])
features1 = scaler.transform(features)
features_transform = pd.DataFrame(features1 , columns = [age, sex, albumin, alkaline_phosphatase, alanine_aminotransferase,
                      aspartate_aminotransferase, bilirubin, cholinesterase, cholesterol,
                      creatinina, gamma_glutamyl_transferase, protein] )


# Add a large "Predict" button with inline HTML styling for size
predict_button = st.markdown(
    """<style>
       .stButton>button {
           padding: 0.75em 1.5em;
           font-size: 1.2em;
       }
       </style>""",
    unsafe_allow_html=True
)

if st.button("Predict"):
    prediction = model.predict(features_transform)[0]
    category, description, precautions, image_path = category_info[prediction]

    # Display the prediction result
    st.subheader("Prediction Result")
    st.write(f"### {category}")
    st.write(f"**Description:** {description}")
    st.write(f"**Precautions:** {precautions}")

    # Display the category-specific image from the local path
    st.image(image_path, caption=category, use_container_width=True)
