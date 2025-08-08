"""
Tests for Stencil AI FastAPI endpoints

Author: Stencil AI Team
Date: 2025-08-07
Dependencies: pytest, httpx, Pillow
"""

import io
import os
import json
import asyncio
import pytest
from PIL import Image
from httpx import AsyncClient

from app.main import app


@pytest.mark.asyncio
async def test_styles_ok():
    async with AsyncClient(app=app, base_url="http://test") as client:
        resp = await client.get("/api/v1/stencils/styles")
        assert resp.status_code == 200
        payload = resp.json()
        assert "styles" in payload and isinstance(payload["styles"], list)
        assert "default_style" in payload and isinstance(payload["default_style"], str)
        assert payload["default_style"] in payload["styles"]


def _make_dummy_png_bytes(size: int = 32) -> bytes:
    img = Image.new("RGB", (size, size), color=(123, 77, 201))
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


@pytest.mark.asyncio
async def test_generate_and_download_flow(tmp_path):
    # Prepare dummy image bytes
    img_bytes = _make_dummy_png_bytes()

    async with AsyncClient(app=app, base_url="http://test") as client:
        # Generate stencil
        files = {"image": ("dummy.png", img_bytes, "image/png")}
        data = {"style": "traditional", "intensity": "0.3", "user_id": "testuser"}
        resp = await client.post("/api/v1/stencils/generate", files=files, data=data)
        assert resp.status_code == 200
        payload = resp.json()
        assert payload.get("status") == "success"
        assert "download_url" in payload

        # Download generated stencil
        dl_url = payload["download_url"]
        resp2 = await client.get(dl_url)
        assert resp2.status_code == 200
        assert resp2.headers.get("content-type") == "image/png"
        assert len(resp2.content) > 0


@pytest.mark.asyncio
async def test_download_invalid_filename_rejected():
    async with AsyncClient(app=app, base_url="http://test") as client:
        # Not a .png extension
        resp = await client.get("/api/v1/stencils/download/invalid.txt")
        assert resp.status_code in (400, 404)

        # Suspicious name even with .png
        resp2 = await client.get("/api/v1/stencils/download/..invalid.png")
        # Router may 404 before validation; both are acceptable as rejection
        assert resp2.status_code in (400, 404)




