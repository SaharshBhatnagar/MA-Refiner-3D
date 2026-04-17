import streamlit as st
import torch
import numpy as np
import plotly.graph_objects as go
from PIL import Image
from torchvision import transforms
from model import Pix2VoxWithMAR
import os

st.set_page_config(page_title="3D Reconstruction UI", layout="wide")
st.title("Multi-Head Attention Refiner for 3D Reconstruction")

@st.cache_resource
def load_model():
    model = Pix2VoxWithMAR()
    if os.path.exists("weights/model_latest.pth"):
        model.load_state_dict(torch.load("weights/model_latest.pth", map_location="cpu", weights_only=True))
        st.sidebar.success("Loaded trained model weights.")
    else:
        st.sidebar.warning("No trained weights found. Using untrained model.")
    model.eval()
    return model

model = load_model()

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

uploaded_files = st.file_uploader("Upload Multi-View Images (Recommended: 8 to 20)", accept_multiple_files=True, type=['png', 'jpg'])

if uploaded_files:
    max_views = 20
    files_to_process = uploaded_files[:max_views]
    
    image_tensors = []
    num_cols = min(len(files_to_process), 5)
    cols = st.columns(num_cols)
    
    for i, file in enumerate(files_to_process):
        img = Image.open(file).convert('RGB')
        cols[i % num_cols].image(img, caption=f"View {i+1}", use_container_width=True)
        image_tensors.append(transform(img))
        
    if st.button("Generate 3D Model"):
        with st.spinner("Processing through Encoder, Decoder, and MA-R..."):
            input_tensor = torch.stack(image_tensors).unsqueeze(0)
            
            with torch.no_grad():
                voxel_probs = model(input_tensor)
                
            voxel_grid = voxel_probs[0, 0].numpy()
            binary_voxels = (voxel_grid > 0.3).astype(np.float32)

            x, y, z = np.where(binary_voxels == 1)
            
            fig = go.Figure(data=[go.Scatter3d(
                x=x, y=y, z=z,
                mode='markers',
                
                marker=dict(size=6, color=z, colorscale='magma', opacity=0.9, symbol='square')
            )])

            fig.update_layout(
                scene=dict(
                    xaxis=dict(range=[0, 32], title='X', visible=False),
                    yaxis=dict(range=[0, 32], title='Y', visible=False),
                    zaxis=dict(range=[0, 32], title='Z', visible=False),
                    aspectmode='cube'
                ),
                margin=dict(l=0, r=0, b=0, t=0),
                height=600
            )

            st.success("Reconstruction Complete")
            st.plotly_chart(fig, use_container_width=True)

