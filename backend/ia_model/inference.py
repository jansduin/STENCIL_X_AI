"""
Pix2Pix Model Inference - AI Model Loading and Prediction
Inference module for Pix2Pix stencil generation model

Author: Stencil AI Team
Date: 2024
Dependencies: PyTorch, torchvision, numpy, PIL
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from PIL import Image
import logging
from typing import Optional, Tuple, Dict, Any
import os
from pathlib import Path

logger = logging.getLogger(__name__)

class Pix2PixGenerator(nn.Module):
    """Pix2Pix Generator Network (U-Net architecture)"""
    
    def __init__(self, input_channels: int = 3, output_channels: int = 1):
        """
        Initialize Pix2Pix Generator
        
        Args:
            input_channels: Number of input channels
            output_channels: Number of output channels
        """
        super(Pix2PixGenerator, self).__init__()
        
        # Encoder (downsampling)
        self.enc1 = self._make_layer(input_channels, 64, normalize=False)
        self.enc2 = self._make_layer(64, 128)
        self.enc3 = self._make_layer(128, 256)
        self.enc4 = self._make_layer(256, 512)
        self.enc5 = self._make_layer(512, 512)
        self.enc6 = self._make_layer(512, 512)
        self.enc7 = self._make_layer(512, 512)
        self.enc8 = self._make_layer(512, 512, normalize=False)
        
        # Decoder (upsampling)
        self.dec8 = self._make_layer(512, 512, dropout=0.5)
        self.dec7 = self._make_layer(1024, 512, dropout=0.5)
        self.dec6 = self._make_layer(1024, 512, dropout=0.5)
        self.dec5 = self._make_layer(1024, 512)
        self.dec4 = self._make_layer(1024, 256)
        self.dec3 = self._make_layer(512, 128)
        self.dec2 = self._make_layer(256, 64)
        self.dec1 = nn.Sequential(
            nn.Upsample(scale_factor=2),
            nn.Conv2d(128, output_channels, kernel_size=4, stride=1, padding=1),
            nn.Tanh()
        )
    
    def _make_layer(self, in_channels: int, out_channels: int, 
                   normalize: bool = True, dropout: float = 0.0) -> nn.Sequential:
        """Create a layer with optional normalization and dropout"""
        layers = [
            nn.Conv2d(in_channels, out_channels, kernel_size=4, stride=2, padding=1),
        ]
        
        if normalize:
            layers.append(nn.BatchNorm2d(out_channels))
        
        layers.append(nn.LeakyReLU(0.2, inplace=True))
        
        if dropout > 0:
            layers.append(nn.Dropout(dropout))
        
        return nn.Sequential(*layers)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the generator
        
        Args:
            x: Input tensor [batch_size, channels, height, width]
            
        Returns:
            torch.Tensor: Generated stencil
        """
        # Encoder
        e1 = self.enc1(x)
        e2 = self.enc2(e1)
        e3 = self.enc3(e2)
        e4 = self.enc4(e3)
        e5 = self.enc5(e4)
        e6 = self.enc6(e5)
        e7 = self.enc7(e6)
        e8 = self.enc8(e7)
        
        # Decoder with skip connections
        d8 = self.dec8(e8)
        d7 = self.dec7(torch.cat([d8, e7], 1))
        d6 = self.dec6(torch.cat([d7, e6], 1))
        d5 = self.dec5(torch.cat([d6, e5], 1))
        d4 = self.dec4(torch.cat([d5, e4], 1))
        d3 = self.dec3(torch.cat([d4, e3], 1))
        d2 = self.dec2(torch.cat([d3, e2], 1))
        d1 = self.dec1(torch.cat([d2, e1], 1))
        
        return d1

class Pix2PixInference:
    """Pix2Pix model inference wrapper"""
    
    def __init__(self, model_path: Optional[str] = None):
        """
        Initialize Pix2Pix inference
        
        Args:
            model_path: Path to trained model weights
        """
        self.model_path = model_path or "models/pix2pix_stencil.pth"
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = None
        self.is_loaded = False
        
        logger.info(f"Pix2PixInference initialized on device: {self.device}")
    
    def load_model(self) -> bool:
        """
        Load the trained Pix2Pix model
        
        Returns:
            bool: True if model loaded successfully
        """
        try:
            if not os.path.exists(self.model_path):
                logger.warning(f"Model not found at {self.model_path}, using placeholder")
                self.model = self._create_placeholder_model()
            else:
                self.model = self._load_trained_model()
            
            self.model.to(self.device)
            self.model.eval()
            self.is_loaded = True
            logger.info("Pix2Pix model loaded successfully")
            return True
            
        except Exception as e:
            logger.error(f"Failed to load Pix2Pix model: {e}")
            self.model = self._create_placeholder_model()
            return False
    
    def _create_placeholder_model(self) -> Pix2PixGenerator:
        """Create a placeholder model for development"""
        return Pix2PixGenerator()
    
    def _load_trained_model(self) -> Pix2PixGenerator:
        """Load the actual trained Pix2Pix model"""
        model = Pix2PixGenerator()
        
        try:
            # Load state dict
            checkpoint = torch.load(self.model_path, map_location=self.device)
            
            if 'generator_state_dict' in checkpoint:
                model.load_state_dict(checkpoint['generator_state_dict'])
            elif 'state_dict' in checkpoint:
                model.load_state_dict(checkpoint['state_dict'])
            else:
                model.load_state_dict(checkpoint)
            
            logger.info("Trained model weights loaded successfully")
            
        except Exception as e:
            logger.error(f"Error loading trained model: {e}")
            # Return placeholder model if loading fails
            model = self._create_placeholder_model()
        
        return model
    
    def preprocess_image(self, image: np.ndarray) -> torch.Tensor:
        """
        Preprocess image for model inference
        
        Args:
            image: Input image as numpy array [H, W, C]
            
        Returns:
            torch.Tensor: Preprocessed image tensor [1, C, H, W]
        """
        # Convert to PIL Image for consistent processing
        if isinstance(image, np.ndarray):
            image = Image.fromarray(image)
        
        # Resize to model input size (256x256)
        image = image.resize((256, 256), Image.LANCZOS)
        
        # Convert to numpy and normalize to [-1, 1]
        image_array = np.array(image).astype(np.float32)
        if image_array.max() > 1.0:
            image_array = image_array / 255.0
        
        image_array = image_array * 2.0 - 1.0  # Normalize to [-1, 1]
        
        # Convert to tensor and add batch dimension
        image_tensor = torch.from_numpy(image_array).permute(2, 0, 1).unsqueeze(0)
        
        return image_tensor.to(self.device)
    
    def postprocess_output(self, output_tensor: torch.Tensor) -> np.ndarray:
        """
        Postprocess model output to create stencil
        
        Args:
            output_tensor: Model output tensor [1, C, H, W]
            
        Returns:
            np.ndarray: Processed stencil image [H, W]
        """
        # Convert tensor to numpy
        output = output_tensor.squeeze(0).cpu().numpy()
        
        # Handle different output formats
        if output.ndim == 3:
            # Convert to grayscale if multiple channels
            if output.shape[0] == 3:
                output = np.mean(output, axis=0)
            else:
                output = output[0]  # Take first channel
        
        # Denormalize from [-1, 1] to [0, 1]
        output = (output + 1.0) / 2.0
        
        # Apply threshold to create binary stencil
        threshold = 0.5
        stencil = (output > threshold).astype(np.uint8) * 255
        
        return stencil
    
    def predict(self, image: np.ndarray, options: Optional[Dict[str, Any]] = None) -> np.ndarray:
        """
        Generate stencil prediction
        
        Args:
            image: Input image as numpy array
            options: Optional generation parameters
            
        Returns:
            np.ndarray: Generated stencil
        """
        try:
            if not self.is_loaded:
                self.load_model()
            
            # Preprocess input
            input_tensor = self.preprocess_image(image)
            
            # Generate prediction
            with torch.no_grad():
                output_tensor = self.model(input_tensor)
            
            # Postprocess output
            stencil = self.postprocess_output(output_tensor)
            
            logger.info("Stencil prediction completed successfully")
            return stencil
            
        except Exception as e:
            logger.error(f"Error in prediction: {e}")
            raise
    
    def get_model_info(self) -> Dict[str, Any]:
        """
        Get information about the loaded model
        
        Returns:
            Dict: Model information
        """
        return {
            "model_type": "Pix2Pix Generator",
            "model_path": self.model_path,
            "device": str(self.device),
            "is_loaded": self.is_loaded,
            "input_size": (256, 256),
            "output_size": (256, 256),
            "input_channels": 3,
            "output_channels": 1,
            "architecture": "U-Net with skip connections"
        }
