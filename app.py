"""FastAPI app for HF Spaces (Docker SDK)."""
import json
from pathlib import Path
from fastapi import FastAPI, Response
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

app = FastAPI()

# Load pre-generated data
with open("coords.json") as f:
    data = json.load(f)
print(f"Loaded {len(data['repos'])} repos")

@app.get("/")
async def index():
    html = Path("frontend/index.html").read_text()
    return HTMLResponse(html)

@app.get("/coords")
async def coords():
    return JSONResponse(data)

@app.get("/app.js")
async def app_js():
    js = Path("frontend/app.js").read_text()
    return Response(content=js, media_type="application/javascript")

@app.get("/style.css")
async def style_css():
    css = Path("frontend/style.css").read_text()
    return Response(content=css, media_type="text/css")
