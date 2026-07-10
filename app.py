import os
import asyncio
import nest_asyncio
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from api.core.config import OUTPUT_DIR, UPLOAD_DIR
from api.core.database import _init_jobs_store
from api.core.worker import job_worker
from api.routes import jobs, editor, search, social, render, subtitle, transcript

# Allow nested event loops where supported. Colab's uvicorn can run on uvloop,
# which is not patchable by nest_asyncio and would fail during app import.
try:
    nest_asyncio.apply()
except ValueError as exc:
    if "uvloop" not in str(exc).lower():
        raise

app = FastAPI(title="OpenShorts API", version="1.0.0")

# 1. Static Files & Directories
os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)
app.mount("/videos", StaticFiles(directory=OUTPUT_DIR), name="videos")

# 2. CORS Configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 3. Register Routers
app.include_router(jobs.router)
app.include_router(editor.router)
app.include_router(search.router)
app.include_router(social.router)
app.include_router(render.router)
app.include_router(subtitle.router)
app.include_router(transcript.router)

@app.get("/api/fonts")
async def list_fonts():
    font_specs = [
        ("Montserrat", "Montserrat", "dashboard/public/fonts/Montserrat-Bold.ttf"),
        ("Anton", "Anton", "dashboard/public/fonts/Anton-Regular.ttf"),
        ("Archivo Black", "Archivo Black", "dashboard/public/fonts/ArchivoBlack-Regular.ttf"),
        ("Bebas Neue", "Bebas Neue", "dashboard/public/fonts/BebasNeue-Regular.ttf"),
        ("Oswald", "Oswald", "dashboard/public/fonts/Oswald-Variable.ttf"),
        ("Teko", "Teko", "dashboard/public/fonts/Teko-Variable.ttf"),
        ("Arial", "Arial", None),
        ("Verdana", "Verdana", None),
    ]
    fonts = [
        {"value": value, "label": label, "available": True if path is None else os.path.exists(path)}
        for value, label, path in font_specs
    ]
    return {"fonts": fonts}

@app.on_event("startup")
async def startup_event():
    # Initialize DB and Background Worker
    _init_jobs_store()
    asyncio.create_task(job_worker())
    print("🚀 OpenShorts Backend Started (Modular Mode)")

@app.get("/")
async def root():
    return {"message": "OpenShorts API is running", "mode": "modular"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
