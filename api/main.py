"""FastAPI server to serve coords.json."""
import json
from pathlib import Path
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

app = FastAPI(title="whoami-dataviz API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["GET"],
    allow_headers=["*"],
)

COORDS_PATH = Path("coords.json")


@app.get("/coords")
async def get_coords():
    """Return the coordinates JSON."""
    if not COORDS_PATH.exists():
        raise HTTPException(status_code=404, detail="coords.json not found. Run the pipeline first.")

    with open(COORDS_PATH) as f:
        return json.load(f)


@app.get("/health")
async def health():
    """Health check endpoint."""
    return {"status": "ok"}


# Mount frontend static files if they exist
frontend_path = Path("frontend")
if frontend_path.exists():
    app.mount("/", StaticFiles(directory="frontend", html=True), name="frontend")
