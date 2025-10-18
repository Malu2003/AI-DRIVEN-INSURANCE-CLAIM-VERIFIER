import streamlit as st
import requests
from PIL import Image
import io
import base64
import json

st.set_page_config(page_title="Medical Claim Verification", layout="wide")

st.title("🏥 Medical Claim Verification System")

# Sidebar for file uploads
with st.sidebar:
    st.header("Upload Documents")
    images = st.file_uploader("Upload Medical Images", type=['png', 'jpg', 'jpeg'], accept_multiple_files=True)
    documents = st.file_uploader("Upload Claim Documents", type=['pdf', 'txt'], accept_multiple_files=True)
    
    if st.button("Analyze Claim"):
        if not images and not documents:
            st.error("Please upload at least one document or image")
        else:
            with st.spinner("Analyzing..."):
                # Process each image
                if images:
                    st.subheader("Image Analysis Results")
                    for img in images:
                        files = {'file': img}
                        try:
                            response = requests.post('http://localhost:5000/api/forgery', files=files)
                            if response.ok:
                                result = response.json()
                                st.write(f"Forgery Probability: {result['forgery_probability']:.2%}")
                                
                                if result['ela_image']:
                                    st.image(base64.b64decode(result['ela_image']), caption="ELA Analysis")
                                
                                if result['highlighted_regions']:
                                    st.image(base64.b64decode(result['highlighted_regions']), 
                                           caption="Suspicious Regions")
                        except:
                            st.error(f"Failed to analyze image: {img.name}")

                # Process text documents
                if documents:
                    st.subheader("Document Analysis Results")
                    for doc in documents:
                        files = {'file': doc}
                        try:
                            response = requests.post('http://localhost:5000/api/icd', files=files)
                            if response.ok:
                                result = response.json()
                                st.write("ICD Code Analysis:", result)
                        except:
                            st.error(f"Failed to analyze document: {doc.name}")

# Main content area
col1, col2 = st.columns(2)

with col1:
    st.header("How It Works")
    st.write("""
    1. Upload medical images and claim documents
    2. Our AI analyzes:
        - Image manipulation detection
        - ICD code validation
        - Risk assessment
    3. Get instant verification results
    """)

with col2:
    st.header("Results Dashboard")
    # Placeholder for results visualization
    if 'latest_result' not in st.session_state:
        st.info("Upload documents to see analysis results")
    else:
        # Show latest analysis results
        pass