import streamlit as st
import requests
import json
from PIL import Image
import io
import pandas as pd
from rdkit import Chem
from rdkit.Chem import Draw

st.set_page_config(page_title="Chemical Structure Analyzer", layout="wide")

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        margin-bottom: 1rem;
    }
    .section-header {
        font-size: 1.5rem;
        font-weight: bold;
        margin-top: 1rem;
        margin-bottom: 0.5rem;
    }
    .info-box {
        padding: 1rem;
        border-radius: 0.5rem;
        background-color: #f0f2f6;
        margin-bottom: 1rem;
    }
    .smile-box {
        background-color: #e6f3ff;
        padding: 0.8rem;
        border-radius: 0.5rem;
        font-family: monospace;
        margin-bottom: 1rem;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 2rem;
    }
    .stTabs [data-baseweb="tab"] {
        height: 4rem;
    }
    .compound-card {
        background-color: white;
        border-radius: 0.5rem;
        padding: 1rem;
        box-shadow: 0 1px 3px rgba(0,0,0,0.12), 0 1px 2px rgba(0,0,0,0.24);
        margin-bottom: 1rem;
    }
</style>
""", unsafe_allow_html=True)

# Application title
st.markdown('<div class="main-header">Chemical Structure Analyzer</div>', unsafe_allow_html=True)
st.write("Upload a chemical structure image to analyze and retrieve compound information.")

# Function to render SMILES as molecule image
def render_molecule(smiles):
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol:
            img = Draw.MolToImage(mol, size=(300, 200))
            return img
        return None
    except:
        return None

# Function to call the API
def process_image(image_file):
    try:
        url = "http://localhost:5001/api/process-image"
        files = {"image": image_file}
        response = requests.post(url, files=files)
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        st.error(f"API Error: {str(e)}")
        return None

# Main layout with two columns
col1, col2 = st.columns([1, 1.5])

with col1:
    st.markdown('<div class="section-header">Upload Chemical Structure Image</div>', unsafe_allow_html=True)
    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])
    
    # Display the uploaded image
    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        st.image(image, caption="Uploaded Image", use_column_width=True)
        
        # Process button
        if st.button("Analyze Structure"):
            with st.spinner("Processing image..."):
                # Reset the file pointer
                uploaded_file.seek(0)
                
                # Call the API
                result = process_image(uploaded_file)
                
                # Store the result in session state
                if result:
                    st.session_state.api_result = result
                    st.success("Analysis complete!")
                else:
                    st.error("Failed to process the image. Please try again.")

# Right column for displaying API results
with col2:
    st.markdown('<div class="section-header">Analysis Results</div>', unsafe_allow_html=True)
    
    # Check if we have results to display
    if 'api_result' in st.session_state and st.session_state.api_result:
        result = st.session_state.api_result
        
        # Display detected SMILE
        st.markdown('<div class="section-header">Detected Structure (SMILES)</div>', unsafe_allow_html=True)
        st.markdown(f'<div class="smile-box">{result["detectedSmile"]}</div>', unsafe_allow_html=True)
        
        # Render the detected molecule
        # detected_mol_img = render_molecule(result["detectedSmile"])
        # if detected_mol_img:
        #     st.image(detected_mol_img, caption="Rendered Structure", use_column_width=True)
        
        # Create tabs for different result sections
        tabs = st.tabs(["Current Compound", "Similar Compounds", "Patents"])
        
        # Current compound tab
        with tabs[0]:
            if "currentCompound" in result["pubchemResults"]:
                current = result["pubchemResults"]["currentCompound"]
                st.markdown('<div class="info-box">', unsafe_allow_html=True)
                st.markdown(f"**Compound ID:** {current['cid']}")
                st.markdown(f"**Name:** {current['recordTitle']}")
                st.markdown(f"**SMILES:** {current['smile']}")
                st.markdown('</div>', unsafe_allow_html=True)
            else:
                st.info("No current compound information available.")
        
        # Similar compounds tab
        with tabs[1]:
            if "similarCompound" in result["pubchemResults"] and result["pubchemResults"]["similarCompound"]:
                for i, compound in enumerate(result["pubchemResults"]["similarCompound"]):
                    st.markdown(f'<div class="compound-card">', unsafe_allow_html=True)
                    col_a, col_b = st.columns([2, 1])
                    
                    with col_a:
                        st.markdown(f"**Name:** {compound['recordTitle']}")
                        st.markdown(f"**Compound ID:** {compound['cid']}")
                        st.markdown(f"**IUPAC Name:** {compound.get('iupacName', 'N/A')}")
                        st.markdown(f"**SMILES:** {compound['smile']}")
                    
                    with col_b:
                        mol_img = render_molecule(compound['smile'])
                        if mol_img:
                            st.image(mol_img, use_column_width=True)
                    
                    st.markdown('</div>', unsafe_allow_html=True)
            else:
                st.info("No similar compounds found.")
        
        # Patents tab
        with tabs[2]:
            if "patents" in result["pubchemResults"] and result["pubchemResults"]["patents"]:
                patent_data = []
                for patent in result["pubchemResults"]["patents"]:
                    patent_data.append({
                        "Patent ID": patent['patent_id'],
                        "Country": patent['country_code'],
                        "Kind Code": patent['kind_code'],
                        "URL": f"[View Patent]({patent['url']})"
                    })
                
                patent_df = pd.DataFrame(patent_data)
                st.dataframe(patent_df, use_container_width=True, hide_index=True)
            else:
                st.info("No patent information available.")
    else:
        st.info("Upload an image and click 'Analyze Structure' to see results here.")

# Footer with instructions
st.markdown("---")
st.markdown("""
**How to use this application:**
1. Upload a chemical structure image using the file uploader on the left
2. Click the 'Analyze Structure' button to process the image
3. View the analysis results on the right, including the detected SMILES, compound information, and similar compounds
""")