import streamlit as st
import requests
from PIL import Image

# Set up Streamlit app
st.set_page_config(page_title="Smile Predictor", layout="wide")

# App header
st.markdown("""
    <h1 style='text-align: center;'>SMILE Predictor</h1>
    <p style='text-align: center;'>Upload an image to predict the SMILES structure</p>
""", unsafe_allow_html=True)

# Function to call Flask API
def predict_smile(image_file):
    try:
        url = "http://localhost:5001/api/predict-smile"  # Flask endpoint
        files = {"image": image_file}
        response = requests.post(url, files=files)
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        st.error(f"API Error: {str(e)}")
        return None

# Upload image section
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Display uploaded image
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_column_width=True)
    
    # Predict button
    if st.button("Predict SMILE"):
        with st.spinner("Predicting..."):
            uploaded_file.seek(0)  # Reset file pointer
            result = predict_smile(uploaded_file)
            
            if result:
                if result.get("success"):
                    st.success("Prediction complete!")
                    st.write("### Predicted SMILE:")
                    st.code(result.get("predictedSmile"), language='text')
                else:
                    st.error(result.get("error"))
            else:
                st.error("Failed to get prediction. Please try again.")

# Footer
st.markdown("---")
st.markdown("""
**How to use:**
1. Upload an image using the file uploader.
2. Click **'Predict SMILE'** to process the image.
3. View the predicted SMILES structure.
""")
