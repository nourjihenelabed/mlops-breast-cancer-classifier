"""
Streamlit Frontend for Breast Cancer Classification
"""
import streamlit as st
import requests
import pandas as pd
import numpy as np

# API endpoint - change this when deploying
API_URL = "http://localhost:8000"  # For local testing

# Page config
st.set_page_config(
    page_title="Breast Cancer Classifier",
    page_icon="🔬",
    layout="wide"
)

# Title
st.title("🔬 Breast Cancer Classification")
st.markdown("### AI-Powered Diagnosis Assistant")
st.divider()

# Sidebar
with st.sidebar:
    st.header("ℹ️ About")
    st.write("""
    This app uses a trained MLP neural network to classify 
    breast cancer tumors as **Benign** or **Malignant** based 
    on 30 numerical features.
    """)
    
    st.divider()
    
    # API Status
    st.subheader("🔌 API Status")
    try:
        health = requests.get(f"{API_URL}/health", timeout=2)
        if health.status_code == 200:
            st.success("✅ Connected")
        else:
            st.error("❌ API Error")
    except:
        st.error("❌ API Offline")
        st.warning("Make sure to run: `uvicorn api.app:app --reload`")

# Main content
tab1, tab2, tab3 = st.tabs(["📊 Manual Input", "📁 File Upload", "📈 Model Info"])

# ========================================
# TAB 1: MANUAL INPUT
# ========================================
with tab1:
    st.subheader("Enter Feature Values")
    st.info("💡 Tip: Use sample data button to see an example")
    
    # Sample data button
    if st.button("📋 Load Sample Data (Malignant)"):
        st.session_state.sample_loaded = True
    
    # Feature names (simplified)
    feature_names = [
        "radius_mean", "texture_mean", "perimeter_mean", "area_mean",
        "smoothness_mean", "compactness_mean", "concavity_mean",
        "concave_points_mean", "symmetry_mean", "fractal_dimension_mean",
        "radius_se", "texture_se", "perimeter_se", "area_se",
        "smoothness_se", "compactness_se", "concavity_se",
        "concave_points_se", "symmetry_se", "fractal_dimension_se",
        "radius_worst", "texture_worst", "perimeter_worst", "area_worst",
        "smoothness_worst", "compactness_worst", "concavity_worst",
        "concave_points_worst", "symmetry_worst", "fractal_dimension_worst"
    ]
    
    # Sample values
    sample_values = [
        17.99, 10.38, 122.8, 1001.0, 0.1184, 0.2776, 0.3001, 0.1471,
        0.2419, 0.07871, 1.095, 0.9053, 8.589, 153.4, 0.006399,
        0.04904, 0.05373, 0.01587, 0.03003, 0.006193, 25.38, 17.33,
        184.6, 2019.0, 0.1622, 0.6656, 0.7119, 0.2654, 0.4601, 0.1189
    ]
    
    # Create input fields
    features = []
    cols = st.columns(3)
    
    for i, (name, default_val) in enumerate(zip(feature_names, sample_values)):
        col = cols[i % 3]
        
        if 'sample_loaded' in st.session_state and st.session_state.sample_loaded:
            value = col.number_input(
                name, 
                value=float(default_val),
                format="%.6f",
                key=f"feature_{i}"
            )
        else:
            value = col.number_input(
                name, 
                value=0.0,
                format="%.6f",
                key=f"feature_{i}"
            )
        features.append(value)
    
    st.divider()
    
    # Predict button
    col1, col2, col3 = st.columns([1, 1, 1])
    with col2:
        predict_button = st.button("🔮 Predict", type="primary", use_container_width=True)
    
    if predict_button:
        if sum(features) == 0:
            st.error("⚠️ Please enter feature values or load sample data")
        else:
            with st.spinner("Making prediction..."):
                try:
                    response = requests.post(
                        f"{API_URL}/predict",
                        json={"features": features},
                        timeout=10
                    )
                    
                    if response.status_code == 200:
                        result = response.json()
                        
                        # Display results
                        st.success("✅ Prediction Complete!")
                        
                        col1, col2, col3 = st.columns(3)
                        
                        with col1:
                            st.metric("Diagnosis", result['diagnosis'])
                        
                        with col2:
                            st.metric("Confidence", f"{result['probability']*100:.2f}%")
                        
                        with col3:
                            prediction_class = "Malignant (1)" if result['prediction'] == 1 else "Benign (0)"
                            st.metric("Class", prediction_class)
                        
                        # Visual indicator
                        if result['diagnosis'] == "Malignant":
                            st.error("⚠️ **High Risk**: Further medical examination recommended")
                        else:
                            st.success("✅ **Low Risk**: Tumor appears benign")
                        
                        # Probability bar
                        st.progress(result['probability'])
                        
                    else:
                        st.error(f"API Error: {response.status_code}")
                        st.json(response.json())
                        
                except requests.exceptions.ConnectionError:
                    st.error("❌ Cannot connect to API. Make sure it's running!")
                except Exception as e:
                    st.error(f"Error: {str(e)}")

# ========================================
# TAB 2: FILE UPLOAD
# ========================================
with tab2:
    st.subheader("Upload CSV File")
    st.write("Upload a CSV file with 30 features (one sample per row)")
    
    uploaded_file = st.file_uploader("Choose a CSV file", type=['csv'])
    
    if uploaded_file is not None:
        try:
            df = pd.read_csv(uploaded_file)
            st.write(f"📊 Loaded {len(df)} samples")
            st.dataframe(df.head())
            
            if st.button("🔮 Predict All", type="primary"):
                with st.spinner("Making predictions..."):
                    try:
                        samples = df.values.tolist()
                        response = requests.post(
                            f"{API_URL}/predict-batch",
                            json={"samples": samples},
                            timeout=30
                        )
                        
                        if response.status_code == 200:
                            results = response.json()['predictions']
                            
                            # Add results to dataframe
                            df['Prediction'] = [r['prediction'] for r in results]
                            df['Probability'] = [r['probability'] for r in results]
                            df['Diagnosis'] = [r['diagnosis'] for r in results]
                            
                            st.success("✅ Predictions complete!")
                            st.dataframe(df)
                            
                            # Summary
                            st.subheader("📊 Summary")
                            col1, col2 = st.columns(2)
                            with col1:
                                benign_count = sum(1 for r in results if r['diagnosis'] == 'Benign')
                                st.metric("Benign", benign_count)
                            with col2:
                                malignant_count = len(results) - benign_count
                                st.metric("Malignant", malignant_count)
                            
                            # Download
                            csv = df.to_csv(index=False)
                            st.download_button(
                                "📥 Download Results",
                                csv,
                                "predictions.csv",
                                "text/csv"
                            )
                        else:
                            st.error(f"API Error: {response.status_code}")
                    except Exception as e:
                        st.error(f"Error: {str(e)}")
        except Exception as e:
            st.error(f"Error reading file: {str(e)}")

# ========================================
# TAB 3: MODEL INFO
# ========================================
with tab3:
    st.subheader("📈 Model Information")
    
    try:
        response = requests.get(f"{API_URL}/model-info", timeout=5)
        if response.status_code == 200:
            info = response.json()
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.metric("Model Type", info['model_type'])
                st.metric("Input Shape", str(info['input_shape']))
                st.metric("Output Shape", str(info['output_shape']))
            
            with col2:
                st.metric("Total Parameters", f"{info['total_params']:,}")
                st.metric("Number of Layers", info['layers'])
            
            st.json(info)
        else:
            st.error("Could not fetch model info")
    except Exception as e:
        st.error(f"Error: {str(e)}")

# Footer
st.divider()
st.caption("🔬 Breast Cancer Classification | Built with Streamlit + FastAPI + TensorFlow")