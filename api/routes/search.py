from fastapi import APIRouter, HTTPException, Header
from typing import Optional, List, Dict, Any
from ..core.models import ClipSearchRequest
from ..services.search_service import (
    _analyze_query_profile, _relax_query_profile, _normalize_weight_triplet
)
from ..services.ai_service import _embed_texts_with_gemini

router = APIRouter(prefix="/api")

@router.post("/search")
async def search_clips(req: ClipSearchRequest, x_gemini_key: Optional[str] = Header(None, alias="X-Gemini-Key")):
    # Logic to search clips using search_service and ai_service (embeddings)
    return {"results": []}

@router.get("/search/profile")
async def get_search_profile(query: str):
    profile = _analyze_query_profile(query)
    return profile
