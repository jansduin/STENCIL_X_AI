"""
Stencil Engine Service - AI Model Integration
Service for generating tattoo stencils using Pix2Pix model

Author: Stencil AI Team
Date: 2024
Dependencies: PyTorch, OpenCV, Pillow, numpy
"""

import cv2
import numpy as np
from PIL import Image
from io import BytesIO
import torch
import torch.nn as nn
from typing import Optional, Tuple, Dict, Any
import logging
import os
from pathlib import Path
import uuid
from datetime import datetime

logger = logging.getLogger(__name__)

class StencilEngine:
    """AI engine for generating tattoo stencils"""
    
    def __init__(self, model_path: Optional[str] = None):
        """
        Initialize the stencil engine
        
        Args:
            model_path: Path to the trained Pix2Pix model
        """
        self.model_path = model_path or "models/pix2pix_stencil.pth"
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = None
        self.is_loaded = False
        
        # Create output directory
        self.output_dir = Path("outputs/stencils")
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"StencilEngine initialized on device: {self.device}")
    
    def load_model(self) -> bool:
        """
        Load the Pix2Pix model
        
        Returns:
            bool: True if model loaded successfully
        """
        try:
            if not os.path.exists(self.model_path):
                logger.warning(f"Model not found at {self.model_path}, using placeholder")
                self.model = self._create_placeholder_model()
            else:
                self.model = self._load_pix2pix_model()
            
            self.model.to(self.device)
            self.model.eval()
            self.is_loaded = True
            logger.info("Model loaded successfully")
            return True
            
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            self.model = self._create_placeholder_model()
            return False
    
    def _create_placeholder_model(self) -> nn.Module:
        """Create a placeholder model for development"""
        class PlaceholderModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.conv = nn.Conv2d(3, 3, 3, padding=1)
                
            def forward(self, x):
                return torch.sigmoid(self.conv(x))
        
        return PlaceholderModel()
    
    def _load_pix2pix_model(self) -> nn.Module:
        """Load the actual Pix2Pix model"""
        # TODO: Implement actual Pix2Pix model loading
        # This is a placeholder for the real implementation
        return self._create_placeholder_model()

    def _pipeline_simple(self, bgr: np.ndarray, options: Optional[Dict[str, Any]] = None) -> np.ndarray:
        """Outline-focused pipeline. Returns grayscale stencil (black lines on white).

        Args:
            bgr: BGR image array
            options: parameters controlling behavior

        Returns:
            uint8 grayscale stencil image
        """
        opts: Dict[str, Any] = options or {}
        intensity: float = float(opts.get("intensity", 0.5) or 0.5)
        line_thickness: int = int(opts.get("line_thickness", 2) or 2)
        smooth_skin: int = int(opts.get("smooth_skin", 1) or 1)
        target_size: int = int(opts.get("target_size", 1024) or 1024)
        outline_only: bool = bool(int(opts.get("outline_only", 1)))
        min_component_area: int = int(opts.get("min_component_area", 120) or 120)
        edge_method: str = str((opts.get("edge_method") or "auto")).lower()
        skeletonize_opt: bool = bool(int(opts.get("skeletonize", 0)))
        # Tone controls
        levels_black: int = int(opts.get("levels_black", 0) or 0)
        levels_white: int = int(opts.get("levels_white", 100) or 100)
        gamma: float = float(opts.get("gamma", 1.0) or 1.0)
        posterize_levels: int = int(opts.get("posterize_levels", 0) or 0)
        clarity: float = float(opts.get("clarity", 0.0) or 0.0)

        # If explicit sketch method requested, dispatch to dedicated pipeline
        if edge_method == "sketch":
            return self._pipeline_sketch(bgr, options)

        # Resize preserving aspect ratio to target_size on max side
        h, w = bgr.shape[:2]
        scale = target_size / float(max(h, w)) if max(h, w) > target_size else 1.0
        if scale != 1.0:
            bgr = cv2.resize(bgr, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)

        # Grayscale
        gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)

        # Bilateral smoothing
        if smooth_skin > 0:
            d = 5 + 2 * smooth_skin
            sigma_color = 25 + 25 * smooth_skin
            sigma_space = 25 + 25 * smooth_skin
            gray = cv2.bilateralFilter(gray, d=d, sigmaColor=sigma_color, sigmaSpace=sigma_space)

        # Local contrast + tone mapping (levels/posterize/gamma/clarity)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        gray_eq = clahe.apply(gray)
        # Levels: remap 0..255 by clipping blacks/whites
        lb = np.clip(levels_black, 0, 99)
        lw = np.clip(levels_white, 1, 100)
        if lw <= lb:
            lw = lb + 1
        lo = np.percentile(gray_eq, lb)
        hi = np.percentile(gray_eq, lw)
        gray_lvl = np.clip((gray_eq.astype(np.float32) - lo) / max(1e-3, (hi - lo)), 0, 1)
        # Posterize (optional)
        if posterize_levels and posterize_levels > 1:
            q = np.clip(posterize_levels, 2, 8)
            gray_lvl = np.floor(gray_lvl * (q - 1) + 0.5) / (q - 1)
        # Gamma
        gray_lvl = np.clip(gray_lvl, 0, 1) ** np.clip(gamma, 0.2, 3.0)
        # Clarity: unsharp mask
        if clarity and clarity != 0.0:
            blur = cv2.GaussianBlur((gray_lvl*255).astype(np.uint8), (0, 0), 1.0)
            us = cv2.addWeighted((gray_lvl*255).astype(np.uint8), 1 + float(clarity), blur, -float(clarity), 0)
            gray_eq = us
        else:
            gray_eq = (gray_lvl * 255).astype(np.uint8)

        # Edge extraction: Canny + optional XDoG fusion for retratos
        med = float(np.median(gray_eq))
        k = float(np.clip(intensity, 0.0, 1.0))
        band = 0.33 if outline_only else 0.66
        lower = int(max(0, (1.0 - band * (1.0 - k)) * med))
        upper = int(min(255, (1.0 + band * (1.0 - k)) * med * 1.33))
        edges_canny = cv2.Canny(gray_eq, lower, upper, L2gradient=True)

        style_val = str(opts.get("style") or "")
        use_xdog = (edge_method == "xdog") or (edge_method == "auto" and style_val == "portrait_realism")
        # Style-specific defaults (if user didn't request tone ops)
        if style_val == "portrait_realism":
            if posterize_levels == 0:
                posterize_levels = 3
            if levels_black == 0:
                levels_black = 5
            if levels_white == 100:
                levels_white = 95
            if abs(gamma - 1.0) < 1e-3:
                gamma = 0.9
            if clarity == 0.0:
                clarity = 0.4
            # slightly raise min component area to drop skin noise
            if min_component_area < 150:
                min_component_area = 150
        if use_xdog:
            gray_norm = gray_eq.astype(np.float32) / 255.0
            sigma = float(opts.get("xdog_sigma", 0.6))
            kxd = float(opts.get("xdog_k", 1.6))
            phi = float(opts.get("xdog_phi", 50.0 if style_val == "portrait_realism" else 30.0))
            eps = float(opts.get("xdog_eps", -0.01))
            g1 = cv2.GaussianBlur(gray_norm, (0, 0), sigma)
            g2 = cv2.GaussianBlur(gray_norm, (0, 0), sigma * kxd)
            dog = g1 - g2
            xdog = (1.0 + np.tanh(phi * (dog - eps))) / 2.0  # 0..1
            edges_xdog = (xdog < 0.5).astype(np.uint8) * 255
            edges = cv2.max(edges_canny, edges_xdog)
        else:
            edges = edges_canny

        # Otsu binarization
        _, bin_map = cv2.threshold(gray_eq, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

        # Filter small components
        nb, labels, stats, _ = cv2.connectedComponentsWithStats(bin_map, connectivity=8)
        filtered = np.zeros_like(bin_map)
        for i in range(1, nb):
            if int(stats[i, cv2.CC_STAT_AREA]) >= min_component_area:
                filtered[labels == i] = 255

        kernel3 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))

        if outline_only:
            grad = cv2.morphologyEx(filtered, cv2.MORPH_GRADIENT, kernel3)
            combo = cv2.max(grad, edges)
            combo = cv2.morphologyEx(combo, cv2.MORPH_OPEN, kernel3, iterations=1)
            _, outline = cv2.threshold(combo, 0, 255, cv2.THRESH_BINARY)
            # Auto-skeletonize for portrait realism unless explicitly disabled
            do_skel = skeletonize_opt or (style_val == "portrait_realism" and edge_method in ("auto", "xdog"))
            if do_skel:
                outline = self._skeletonize_binary(outline)
            stencil = 255 - outline
        else:
            at = cv2.adaptiveThreshold(gray_eq, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                        cv2.THRESH_BINARY, blockSize=11, C=2)
            at_inv = 255 - at
            combo = cv2.max(edges, at_inv)
            stencil = 255 - cv2.morphologyEx(combo, cv2.MORPH_CLOSE, kernel3, iterations=1)

        # Thickness control (invert->morph->invert)
        thickness = max(1, min(5, int(line_thickness)))
        if thickness != 1:
            ksize = 1 + 2 * (abs(thickness - 1))
            ker = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (ksize, ksize))
            inv = 255 - stencil
            if thickness > 1:
                inv = cv2.dilate(inv, ker, iterations=1)
            else:
                inv = cv2.erode(inv, ker, iterations=1)
            stencil = 255 - inv

        stencil = cv2.bilateralFilter(stencil, d=5, sigmaColor=25, sigmaSpace=25)
        if stencil.ndim == 3:
            stencil = cv2.cvtColor(stencil, cv2.COLOR_BGR2GRAY)

        return stencil

    def _pipeline_sketch(self, bgr: np.ndarray, options: Optional[Dict[str, Any]] = None) -> np.ndarray:
        """Pencil-sketch oriented pipeline focused en bajo ruido y contornos nítidos.

        Pasos: denoise + bilateral/median -> color dodge (gray vs blurred invert) ->
        claridad opcional -> Canny sobre sketch -> limpieza morfológica y control de grosor.
        """
        opts: Dict[str, Any] = options or {}
        intensity: float = float(opts.get("intensity", 0.6) or 0.6)
        line_thickness: int = int(opts.get("line_thickness", 2) or 2)
        smooth_skin: int = int(opts.get("smooth_skin", 1) or 1)
        target_size: int = int(opts.get("target_size", 1024) or 1024)
        outline_only: bool = bool(int(opts.get("outline_only", 1)))
        skeletonize_opt: bool = bool(int(opts.get("skeletonize", 0)))
        denoise_h: int = int(opts.get("denoise_h", 7) or 7)
        clarity: float = float(opts.get("clarity", 0.3) or 0.3)
        sigma: float = float(opts.get("sketch_sigma", 8.0) or 8.0)
        min_component_area: int = int(opts.get("min_component_area", 150) or 150)

        # Resize preserving aspect ratio to target_size on max side
        h, w = bgr.shape[:2]
        scale = target_size / float(max(h, w)) if max(h, w) > target_size else 1.0
        if scale != 1.0:
            bgr = cv2.resize(bgr, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)

        gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
        # Denoise
        if denoise_h > 0:
            gray = cv2.fastNlMeansDenoising(gray, None, h=denoise_h, templateWindowSize=7, searchWindowSize=21)
        # Edge-preserving smoothing
        if smooth_skin > 0:
            d = 5 + 2 * smooth_skin
            gray = cv2.bilateralFilter(gray, d=d, sigmaColor=30 + 20 * smooth_skin, sigmaSpace=30 + 20 * smooth_skin)
        gray = cv2.medianBlur(gray, 3)

        # Color dodge sketch
        inv = 255 - gray
        blur = cv2.GaussianBlur(inv, (0, 0), sigma)
        dodge = cv2.divide(gray, 255 - blur, scale=256)
        sketch = np.clip(dodge, 0, 255).astype(np.uint8)

        # Clarity (unsharp)
        if clarity and abs(clarity) > 1e-6:
            g = cv2.GaussianBlur(sketch, (0, 0), 1.0)
            sketch = cv2.addWeighted(sketch, 1 + float(clarity), g, -float(clarity), 0)

        # Edges sobre sketch y limpieza
        med = float(np.median(sketch))
        k = float(np.clip(intensity, 0.0, 1.0))
        lower = int(max(0, (1.0 - 0.5 * (1.0 - k)) * med))
        upper = int(min(255, (1.0 + 0.5 * (1.0 - k)) * med * 1.33))
        edges = cv2.Canny(sketch, lower, upper, L2gradient=True)
        kernel3 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        edges = cv2.morphologyEx(edges, cv2.MORPH_OPEN, kernel3, iterations=1)

        if outline_only:
            # Remove tiny components
            nb, labels, stats, _ = cv2.connectedComponentsWithStats(edges, connectivity=8)
            mask = np.zeros_like(edges)
            for i in range(1, nb):
                if int(stats[i, cv2.CC_STAT_AREA]) >= min_component_area:
                    mask[labels == i] = 255
            outline = mask
            stencil = 255 - outline
            # Thickness control
            thickness = max(1, min(5, int(line_thickness)))
            if thickness != 1:
                ksize = 1 + 2 * (abs(thickness - 1))
                ker = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (ksize, ksize))
                invw = 255 - stencil
                if thickness > 1:
                    invw = cv2.dilate(invw, ker, iterations=1)
                else:
                    invw = cv2.erode(invw, ker, iterations=1)
                stencil = 255 - invw
            if skeletonize_opt:
                invw = 255 - stencil
                invw = self._skeletonize_binary(invw)
                stencil = 255 - invw
            return stencil
        else:
            return sketch

    def _skeletonize_binary(self, binary_img: np.ndarray, max_iters: int = 100) -> np.ndarray:
        """Morphological skeletonization for a binary image with white foreground (255).

        Args:
            binary_img: uint8 binary image (0/255), white foreground.
            max_iters: safety cap on iterations.

        Returns:
            uint8 binary skeleton (0/255) with white foreground.
        """
        img = (binary_img > 0).astype(np.uint8) * 255
        skel = np.zeros_like(img)
        element = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))
        iters = 0
        while True:
            iters += 1
            eroded = cv2.erode(img, element)
            temp = cv2.dilate(eroded, element)
            temp = cv2.subtract(img, temp)
            skel = cv2.bitwise_or(skel, temp)
            img = eroded.copy()
            if cv2.countNonZero(img) == 0 or iters >= max_iters:
                break
        # Spur pruning: remove isolated endpoints lightly
        kernel = np.array([[1,1,1],[1,10,1],[1,1,1]], dtype=np.uint8)
        for _ in range(2):
            neighbors = cv2.filter2D((skel>0).astype(np.uint8), -1, kernel)
            endpoints = ((neighbors==11) & (skel>0)).astype(np.uint8)*255
            skel = cv2.subtract(skel, endpoints)
        return skel
    
    def preprocess_image(self, image_path: str) -> torch.Tensor:
        """
        Preprocess input image for model inference
        
        Args:
            image_path: Path to input image
            
        Returns:
            torch.Tensor: Preprocessed image tensor
        """
        try:
            # Load image
            image = cv2.imread(image_path)
            if image is None:
                raise ValueError(f"Could not load image from {image_path}")
            
            # Convert BGR to RGB
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            
            # Resize to model input size (256x256)
            image = cv2.resize(image, (256, 256))
            
            # Normalize to [0, 1]
            image = image.astype(np.float32) / 255.0
            
            # Convert to tensor and add batch dimension
            image_tensor = torch.from_numpy(image).permute(2, 0, 1).unsqueeze(0)
            
            return image_tensor.to(self.device)
            
        except Exception as e:
            logger.error(f"Error preprocessing image: {e}")
            raise
    
    def generate_stencil(self, image_path: str, options: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Generate stencil from input image
        
        Args:
            image_path: Path to input image
            options: Generation options (style, intensity, etc.)
            
        Returns:
            Dict containing stencil path and metadata
        """
        try:
            if not self.is_loaded:
                self.load_model()
            
            # Preprocess image
            input_tensor = self.preprocess_image(image_path)
            
            # Generate stencil
            with torch.no_grad():
                output_tensor = self.model(input_tensor)
            
            # Postprocess output
            stencil_image = self._postprocess_output(output_tensor)
            
            # Save stencil
            stencil_path = self._save_stencil(stencil_image, image_path)
            
            # Generate metadata
            metadata = self._generate_metadata(image_path, stencil_path, options)
            
            logger.info(f"Stencil generated successfully: {stencil_path}")
            return metadata
            
        except Exception as e:
            logger.error(f"Error generating stencil: {e}")
            raise

    def generate_stencil_simple(self, image_path: str, options: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Generate stencil using a handcrafted OpenCV pipeline (no ML weights).

        This pipeline is tuned for portraits/realism baseline and aims to produce
        clean line art with controllable thickness and reduced skin noise.

        Args:
            image_path: Path to the input image on disk.
            options: Optional parameters controlling the generation, such as:
                - style: str, e.g., "portrait_realism"
                - intensity: float in [0,1], sensitivity of edges (default 0.5)
                - line_thickness: int, dilation kernel size in [1..5] (default 2)
                - smooth_skin: int, strength of bilateral smoothing [0..2] (default 1)
                - target_size: int, max side resolution (default 1024)

        Returns:
            Dict containing stencil path and metadata.
        """
        try:
            opts: Dict[str, Any] = options or {}
            intensity: float = float(opts.get("intensity", 0.5) or 0.5)
            line_thickness: int = int(opts.get("line_thickness", 2) or 2)
            smooth_skin: int = int(opts.get("smooth_skin", 1) or 1)
            target_size: int = int(opts.get("target_size", 1024) or 1024)
            outline_only: bool = bool(int(opts.get("outline_only", 1)))  # default ON
            min_component_area: int = int(opts.get("min_component_area", 120) or 120)

            # Load image (BGR)
            bgr = cv2.imread(image_path)
            if bgr is None:
                raise ValueError(f"Could not load image from {image_path}")

            stencil = self._pipeline_simple(bgr, options=opts)

            # Save result (transparent background PNG, 300 DPI)
            transparent_bg = bool(int(opts.get("transparent_bg", 1)))
            stencil_path = self._save_stencil_png(stencil, image_path, dpi=300, transparent_bg=transparent_bg)
            metadata = self._generate_metadata(image_path, stencil_path, options)
            logger.info(f"Stencil (simple) generated successfully: {stencil_path}")
            return metadata

        except Exception as e:
            logger.error(f"Error in simple stencil generation: {e}")
            raise
    
    def _postprocess_output(self, output_tensor: torch.Tensor) -> np.ndarray:
        """
        Postprocess model output to create stencil image
        
        Args:
            output_tensor: Model output tensor
            
        Returns:
            np.ndarray: Processed stencil image
        """
        # Convert tensor to numpy
        output = output_tensor.squeeze(0).cpu().numpy()
        output = np.transpose(output, (1, 2, 0))
        
        # Convert to grayscale for stencil
        if output.shape[2] == 3:
            output = np.mean(output, axis=2)
        
        # Apply threshold to create binary stencil
        threshold = 0.5
        stencil = (output > threshold).astype(np.uint8) * 255
        
        return stencil
    
    def _save_stencil(self, stencil_image: np.ndarray, original_path: str) -> str:
        """
        Save generated stencil to file
        
        Args:
            stencil_image: Stencil image array
            original_path: Path to original image
            
        Returns:
            str: Path to saved stencil
        """
        # Generate unique filename
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        unique_id = str(uuid.uuid4())[:8]
        original_name = Path(original_path).stem
        
        filename = f"stencil_{original_name}_{timestamp}_{unique_id}.png"
        stencil_path = self.output_dir / filename
        
        # Save stencil
        cv2.imwrite(str(stencil_path), stencil_image)
        
        return str(stencil_path)

    def _save_stencil_png(self, stencil_image: np.ndarray, original_path: str, dpi: int = 300, transparent_bg: bool = True) -> str:
        """Save stencil as PNG with optional transparent background and DPI metadata.

        Args:
            stencil_image: Grayscale image with black lines on white background.
            original_path: Original input path (for naming).
            dpi: Target DPI metadata to embed in PNG.
            transparent_bg: Convert white background to transparent alpha.

        Returns:
            Path to saved PNG on disk.
        """
        # Generate unique filename
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        unique_id = str(uuid.uuid4())[:8]
        original_name = Path(original_path).stem
        filename = f"stencil_{original_name}_{timestamp}_{unique_id}.png"
        stencil_path = self.output_dir / filename

        # Ensure grayscale uint8
        if stencil_image.ndim == 3:
            stencil_image = cv2.cvtColor(stencil_image, cv2.COLOR_BGR2GRAY)
        arr = np.clip(stencil_image, 0, 255).astype(np.uint8)

        if transparent_bg:
            # White → alpha 0; Non-white → alpha 255
            alpha = np.where(arr > 250, 0, 255).astype(np.uint8)
            rgba = np.dstack([arr, arr, arr, alpha])
            im = Image.fromarray(rgba, mode="RGBA")
            im.save(str(stencil_path), format="PNG", dpi=(dpi, dpi))
        else:
            im = Image.fromarray(arr, mode="L")
            im.save(str(stencil_path), format="PNG", dpi=(dpi, dpi))

        return str(stencil_path)
    
    def render_stencil_simple_from_bytes(self, image_bytes: bytes, options: Optional[Dict[str, Any]] = None) -> bytes:
        """Render preview PNG bytes for the simple pipeline without saving to disk.

        Args:
            image_bytes: Raw image bytes from upload.
            options: Dict of parameters (same as generate_stencil_simple).

        Returns:
            PNG bytes ready to send to client.
        """
        opts: Dict[str, Any] = options or {}
        transparent_bg = bool(int(opts.get("transparent_bg", 1)))

        data = np.frombuffer(image_bytes, dtype=np.uint8)
        bgr = cv2.imdecode(data, cv2.IMREAD_COLOR)
        if bgr is None:
            raise ValueError("Could not decode image bytes")

        stencil = self._pipeline_simple(bgr, options=opts)

        # Encode to PNG with optional alpha
        if transparent_bg:
            alpha = np.where(stencil > 250, 0, 255).astype(np.uint8)
            rgba = np.dstack([stencil, stencil, stencil, alpha])
            img = Image.fromarray(rgba, mode="RGBA")
        else:
            img = Image.fromarray(stencil, mode="L")

        buf = BytesIO()
        img.save(buf, format="PNG", dpi=(150, 150))  # preview DPI lower
        return buf.getvalue()

    def _generate_metadata(self, input_path: str, output_path: str, options: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Generate metadata for the stencil generation
        
        Args:
            input_path: Path to input image
            output_path: Path to generated stencil
            options: Generation options
            
        Returns:
            Dict: Metadata about the generation
        """
        return {
            "input_path": input_path,
            "output_path": output_path,
            "generation_time": datetime.now().isoformat(),
            "model_used": "pix2pix_stencil",
            "options": options or {},
            "status": "success"
        }
    
    def get_model_info(self) -> Dict[str, Any]:
        """
        Get information about the loaded model
        
        Returns:
            Dict: Model information
        """
        return {
            "model_path": self.model_path,
            "device": str(self.device),
            "is_loaded": self.is_loaded,
            "model_type": "Pix2Pix",
            "input_size": (256, 256),
            "output_size": (256, 256)
        }
