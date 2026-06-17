import os
from typing import Any, Dict

import httpx
from fastapi import APIRouter, HTTPException

from api.core.config import RENDER_SERVICE_URL

router = APIRouter(prefix="/api/render", tags=["render"])


def _normalize_output_url(payload: Dict[str, Any]) -> Dict[str, Any]:
    output_url = payload.get("outputUrl")
    if not isinstance(output_url, str) or not output_url:
        return payload

    if output_url.startswith("/output/"):
        payload["outputUrl"] = output_url.replace("/output/", "/videos/", 1)
        return payload

    marker = f"{os.sep}output{os.sep}"
    if marker in output_url:
        relative_path = output_url.split(marker, 1)[1].replace(os.sep, "/")
        payload["outputUrl"] = f"/videos/{relative_path}"

    return payload


@router.get("/health")
async def render_health():
    try:
        async with httpx.AsyncClient(timeout=5.0) as client:
            response = await client.get(f"{RENDER_SERVICE_URL}/health")
            response.raise_for_status()
            return response.json()
    except httpx.HTTPError as exc:
        raise HTTPException(status_code=503, detail=f"Render service unavailable: {exc}") from exc


@router.post("/remotion")
async def submit_remotion_render(payload: Dict[str, Any]):
    try:
        async with httpx.AsyncClient(timeout=15.0) as client:
            response = await client.post(f"{RENDER_SERVICE_URL}/render", json=payload)
    except httpx.HTTPError as exc:
        raise HTTPException(status_code=503, detail=f"Render service unavailable: {exc}") from exc

    if response.status_code >= 400:
        raise HTTPException(status_code=response.status_code, detail=response.text)

    return response.json()


@router.get("/remotion/{render_id}")
async def get_remotion_render(render_id: str):
    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            response = await client.get(f"{RENDER_SERVICE_URL}/render/{render_id}")
    except httpx.HTTPError as exc:
        raise HTTPException(status_code=503, detail=f"Render service unavailable: {exc}") from exc

    if response.status_code >= 400:
        raise HTTPException(status_code=response.status_code, detail=response.text)

    return _normalize_output_url(response.json())
