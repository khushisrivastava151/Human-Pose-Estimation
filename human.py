import streamlit as st
import torch
import torch.nn as nn
import torchvision.transforms as transforms
import cv2
import numpy as np
import matplotlib.pyplot as plt

class PoseEstimationModel(nn.Module):
    def __init__(self, num_joints=17):
        super(PoseEstimationModel, self).__init__()
        
        # Backbone (simplified ResNet-like feature extractor)
        self.backbone = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
            
            # Residual-style blocks
            self._conv_block(64, 128),
            self._conv_block(128, 256),
            self._conv_block(256, 512)
        )
        
        # Pose estimation head
        self.pose_head = nn.Sequential(
            nn.Conv2d(512, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, num_joints, kernel_size=1)
        )
    
    def _conv_block(self, in_channels, out_channels):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
    
    def forward(self, x):
        features = self.backbone(x)
        heatmaps = self.pose_head(features)
        return heatmaps

def load_model(model_path):
    """Load pre-trained model."""
    model = PoseEstimationModel()
    model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
    model.eval()
    return model

def preprocess_image(image):
    """Preprocess image for pose estimation."""
    # Convert to RGB
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # Resize and normalize
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        transforms.Resize((256, 256))
    ])
    
    input_tensor = transform(image_rgb).unsqueeze(0)
    return input_tensor

def detect_keypoints(model, image):
    """Detect pose keypoints."""
    # Preprocess image
    input_tensor = preprocess_image(image)
    
    # Inference
    with torch.no_grad():
        heatmaps = model(input_tensor)
    
    # Process heatmaps
    heatmaps = heatmaps.squeeze().numpy()
    keypoints = []
    
    # Extract keypoint locations
    for heatmap in heatmaps:
        y, x = np.unravel_index(heatmap.argmax(), heatmap.shape)
        keypoints.append((x, y))
    
    return keypoints

def visualize_keypoints(image, keypoints):
    """Visualize detected keypoints."""
    plt.figure(figsize=(10, 10))
    plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    
    # Plot keypoints
    for (x, y) in keypoints:
        plt.scatter(x, y, color='red', s=100)
    
    plt.axis('off')
    plt.tight_layout()
    return plt

def main():
    st.title('Human Pose Estimation App')
    
    # Sidebar for model and upload
    st.sidebar.header('Pose Estimation')
    
    # File uploader
    uploaded_file = st.sidebar.file_uploader(
        "Choose an image...", 
        type=['jpg', 'jpeg', 'png']
    )
    
    # Model selection (placeholder)
    model_type = st.sidebar.selectbox(
        'Select Pose Estimation Model',
        ['Default Model']
    )
    
    # Main content area
    if uploaded_file is not None:
        # Read the image
        file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
        image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
        
        # Display original image
        st.subheader('Original Image')
        st.image(cv2.cvtColor(image, cv2.COLOR_BGR2RGB), use_column_width=True)
        
        # Load model (replace with actual model path)
        try:
            model = load_model('pose_estimation_model.pth')
            
            # Detect keypoints
            keypoints = detect_keypoints(model, image)
            
            # Visualize keypoints
            st.subheader('Pose Estimation Result')
            fig = visualize_keypoints(image, keypoints)
            st.pyplot(fig)
            
            # Display keypoints
            st.subheader('Detected Keypoints')
            keypoint_df = pd.DataFrame(keypoints, columns=['X', 'Y'])
            st.dataframe(keypoint_df)
        
        except Exception as e:
            st.error(f"Error processing image: {e}")
    
    else:
        st.info('Please upload an image to get started.')

if __name__ == '__main__':
    main()

# Requirements file (requirements.txt)
"""
streamlit
torch
torchvision
opencv-python
numpy
matplotlib
pandas
"""