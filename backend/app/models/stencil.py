"""
Stencil Models - Pydantic Schemas
Models for stencil-related API responses and requests

Author: Stencil AI Team
Date: 2025-08-07
Dependencies: Pydantic, typing
"""

from pydantic import BaseModel, Field
from typing import List


class StylesResponse(BaseModel):
    """Response schema for available stencil styles.

    Attributes:
        styles: Allowed style identifiers.
        default_style: Default style identifier.
    """

    styles: List[str] = Field(..., description="Available stencil styles")
    default_style: str = Field(..., description="Default stencil style")




