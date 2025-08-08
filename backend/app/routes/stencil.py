"""
Stencil API Routes - FastAPI Endpoints
API endpoints for stencil generation and management

Author: Stencil AI Team
Date: 2024
Dependencies: FastAPI, UploadFile, File, HTTPException
"""

from fastapi import APIRouter, HTTPException, UploadFile, File, Depends, Form
from fastapi.responses import FileResponse, JSONResponse, Response
from typing import Optional, Dict, Any, List
import os
import logging
from pathlib import Path
import uuid
from datetime import datetime

from ..services.stencil_engine import StencilEngine
from ..models.user import UserResponse
from ..models.stencil import StylesResponse

logger = logging.getLogger(__name__)

# Initialize router
router = APIRouter(prefix="/api/v1/stencils", tags=["stencils"])

# Initialize stencil engine (lazy loading)
_stencil_engine = None

def get_stencil_engine():
    """Get or create stencil engine instance"""
    global _stencil_engine
    if _stencil_engine is None:
        _stencil_engine = StencilEngine()
    return _stencil_engine

@router.post("/generate")
async def generate_stencil(
    image: UploadFile = File(..., description="Input image for stencil generation"),
    style: Optional[str] = Form(None, description="Stencil style preference"),
    intensity: Optional[float] = Form(0.5, ge=0.0, le=1.0, description="Stencil intensity"),
    user_id: Optional[str] = Form(None, description="User ID for tracking"),
    mode: Optional[str] = Form("ai", description="Generation mode: 'simple' or 'ai'"),
    # Advanced params (optional)
    line_thickness: Optional[int] = Form(2),
    smooth_skin: Optional[int] = Form(1),
    target_size: Optional[int] = Form(1024),
    outline_only: Optional[int] = Form(1),
    min_component_area: Optional[int] = Form(120),
    transparent_bg: Optional[int] = Form(1),
    edge_method: Optional[str] = Form("auto"),
    skeletonize: Optional[int] = Form(0),
    sketch_sigma: Optional[float] = Form(8.0),
    denoise_h: Optional[int] = Form(7),
    # Tone controls
    levels_black: Optional[int] = Form(0),
    levels_white: Optional[int] = Form(100),
    gamma: Optional[float] = Form(1.0),
    posterize_levels: Optional[int] = Form(0),
    clarity: Optional[float] = Form(0.0),
    # Export/print
    page_size: Optional[str] = Form("none"),  # none|a4|letter
    stencil_width_cm: Optional[float] = Form(None),
    dpi: Optional[int] = Form(300),
) -> Dict[str, Any]:
    """
    Generate stencil from uploaded image
    
    Args:
        image: Uploaded image file
        style: Optional style preference
        intensity: Stencil intensity (0.0 to 1.0)
        user_id: Optional user ID for tracking
        
    Returns:
        Dict containing stencil information and download URL
    """
    try:
        # Validate file type
        if not image.content_type.startswith("image/"):
            raise HTTPException(status_code=400, detail="File must be an image")
        
        # Create upload directory
        upload_dir = Path("uploads/temp")
        upload_dir.mkdir(parents=True, exist_ok=True)
        
        # Save uploaded image
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        unique_id = str(uuid.uuid4())[:8]
        file_extension = Path(image.filename).suffix
        filename = f"input_{timestamp}_{unique_id}{file_extension}"
        file_path = upload_dir / filename
        
        with open(file_path, "wb") as buffer:
            content = await image.read()
            buffer.write(content)
        
        # Prepare generation options
        options = {
            "style": style,
            "intensity": intensity,
            "user_id": user_id,
            "original_filename": image.filename,
            "mode": mode,
            # Advanced passthrough
            "line_thickness": line_thickness,
            "smooth_skin": smooth_skin,
            "target_size": target_size,
            "outline_only": outline_only,
            "min_component_area": min_component_area,
            "transparent_bg": transparent_bg,
            "edge_method": edge_method,
            "skeletonize": skeletonize,
            "sketch_sigma": sketch_sigma,
            "denoise_h": denoise_h,
            # Tone controls
            "levels_black": levels_black,
            "levels_white": levels_white,
            "gamma": gamma,
            "posterize_levels": posterize_levels,
            "clarity": clarity,
            # Export
            "page_size": page_size,
            "stencil_width_cm": stencil_width_cm,
            "dpi": dpi,
        }
        
        # Generate stencil routing by mode/style
        engine = get_stencil_engine()
        if (mode or "").lower() == "simple":
            result = engine.generate_stencil_simple(str(file_path), options)
        else:
            # mode=ai (default)
            result = engine.generate_stencil(str(file_path), options)
        
        # Prepare response
        response = {
            "stencil_id": str(uuid.uuid4()),
            "original_image": image.filename,
            "stencil_path": result["output_path"],
            "download_url": f"/api/v1/stencils/download/{Path(result['output_path']).name}",
            "generation_time": result["generation_time"],
            "options": options,
            "status": "success"
        }
        
        logger.info(f"Stencil generated successfully for user {user_id}")
        return response
        
    except Exception as e:
        logger.error(f"Error generating stencil: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to generate stencil: {str(e)}")


@router.post("/preview_simple")
async def preview_stencil_simple(
    image: UploadFile = File(..., description="Input image for preview"),
    style: Optional[str] = Form(None),
    intensity: Optional[float] = Form(0.5, ge=0.0, le=1.0),
    line_thickness: Optional[int] = Form(2, ge=1, le=5),
    smooth_skin: Optional[int] = Form(1, ge=0, le=2),
    target_size: Optional[int] = Form(768),
    outline_only: Optional[int] = Form(1),
    min_component_area: Optional[int] = Form(120),
    transparent_bg: Optional[int] = Form(1),
    edge_method: Optional[str] = Form("auto"),
    skeletonize: Optional[int] = Form(0),
    sketch_sigma: Optional[float] = Form(8.0),
    denoise_h: Optional[int] = Form(7),
    # Tone controls
    levels_black: Optional[int] = Form(0),
    levels_white: Optional[int] = Form(100),
    gamma: Optional[float] = Form(1.0),
    posterize_levels: Optional[int] = Form(0),
    clarity: Optional[float] = Form(0.0),
) -> Response:
    """Return PNG preview of the Simple pipeline without saving to disk."""
    try:
        if not image.content_type.startswith("image/"):
            raise HTTPException(status_code=400, detail="File must be an image")

        content = await image.read()
        opts = {
            "style": style,
            "intensity": intensity,
            "line_thickness": line_thickness,
            "smooth_skin": smooth_skin,
            "target_size": target_size,
            "outline_only": outline_only,
            "min_component_area": min_component_area,
            "transparent_bg": transparent_bg,
            "edge_method": edge_method,
            "skeletonize": skeletonize,
            "sketch_sigma": sketch_sigma,
            "denoise_h": denoise_h,
            # Tone controls
            "levels_black": levels_black,
            "levels_white": levels_white,
            "gamma": gamma,
            "posterize_levels": posterize_levels,
            "clarity": clarity,
        }
        png_bytes = get_stencil_engine().render_stencil_simple_from_bytes(content, options=opts)
        return Response(content=png_bytes, media_type="image/png")
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error generating preview: {e}")
        raise HTTPException(status_code=500, detail="Failed to generate preview")

@router.get("/download/{filename}")
async def download_stencil(filename: str) -> FileResponse:
    """
    Download generated stencil
    
    Args:
        filename: Stencil filename
        
    Returns:
        FileResponse: Stencil file
    """
    try:
        # Validate filename strictly: allow alphanumerics, dash, underscore, dot only, and .png extension
        import re
        if not re.fullmatch(r"[A-Za-z0-9._-]+", filename or "") or not filename.lower().endswith(".png"):
            raise HTTPException(status_code=400, detail="Invalid filename")

        base_dir = Path("outputs/stencils").resolve()
        requested_path = (base_dir / filename).resolve()

        # Ensure the resolved path is within the base_dir
        try:
            requested_path.relative_to(base_dir)
        except ValueError:
            raise HTTPException(status_code=400, detail="Invalid path")

        if not requested_path.exists():
            raise HTTPException(status_code=404, detail="Stencil not found")

        return FileResponse(
            path=str(requested_path),
            filename=filename,
            media_type="image/png"
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error downloading stencil: {e}")
        raise HTTPException(status_code=500, detail="Failed to download stencil")

@router.get("/model/info")
async def get_model_info() -> Dict[str, Any]:
    """
    Get information about the AI model
    
    Returns:
        Dict: Model information
    """
    try:
        return get_stencil_engine().get_model_info()
    except Exception as e:
        logger.error(f"Error getting model info: {e}")
        raise HTTPException(status_code=500, detail="Failed to get model information")

@router.get("/styles", response_model=StylesResponse)
async def get_available_styles() -> StylesResponse:
    """
    Get available stencil styles
    
    Returns:
        Dict: Available styles
    """
    return StylesResponse(
        styles=[
            "traditional",
            "minimalist",
            "geometric",
            "organic",
            "tribal",
            "japanese",
            "american_traditional",
            "neo_traditional",
            "portrait_realism",
        ],
        default_style="traditional",
    )

@router.get("/history/{user_id}")
async def get_user_stencils(
    user_id: str,
    page: int = 1,
    size: int = 10
) -> Dict[str, Any]:
    """
    Get user's stencil history
    
    Args:
        user_id: User ID
        page: Page number
        size: Page size
        
    Returns:
        Dict: User's stencil history
    """
    try:
        # TODO: Implement database query for user stencils
        # This is a placeholder response
        return {
            "user_id": user_id,
            "stencils": [],
            "total": 0,
            "page": page,
            "size": size,
            "pages": 0
        }
        
    except Exception as e:
        logger.error(f"Error getting user stencils: {e}")
        raise HTTPException(status_code=500, detail="Failed to get user stencils")

@router.delete("/{stencil_id}")
async def delete_stencil(stencil_id: str) -> Dict[str, str]:
    """
    Delete a stencil
    
    Args:
        stencil_id: Stencil ID to delete
        
    Returns:
        Dict: Deletion confirmation
    """
    try:
        # TODO: Implement stencil deletion logic
        # This is a placeholder response
        return {
            "message": "Stencil deleted successfully",
            "stencil_id": stencil_id
        }
        
    except Exception as e:
        logger.error(f"Error deleting stencil: {e}")
        raise HTTPException(status_code=500, detail="Failed to delete stencil")

@router.post("/batch")
async def generate_batch_stencils(
    images: List[UploadFile] = File(..., description="Multiple images for batch processing"),
    style: Optional[str] = Form(None, description="Stencil style for all images"),
    user_id: Optional[str] = Form(None, description="User ID for tracking")
) -> Dict[str, Any]:
    """
    Generate stencils for multiple images
    
    Args:
        images: List of uploaded images
        style: Style to apply to all images
        user_id: User ID for tracking
        
    Returns:
        Dict: Batch processing results
    """
    try:
        results = []
        
        for image in images:
            # Validate file type
            if not image.content_type.startswith("image/"):
                continue
            
            # Process each image
            # TODO: Implement batch processing logic
            results.append({
                "filename": image.filename,
                "status": "pending",
                "message": "Batch processing not yet implemented"
            })
        
        return {
            "batch_id": str(uuid.uuid4()),
            "total_images": len(images),
            "processed": len(results),
            "results": results,
            "user_id": user_id
        }
        
    except Exception as e:
        logger.error(f"Error in batch processing: {e}")
        raise HTTPException(status_code=500, detail="Failed to process batch")
