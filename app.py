"""Hugging Face Spaces app - serves the visualization."""
import gradio as gr
from pathlib import Path

# Read the frontend files
css_content = Path("frontend/style.css").read_text()
js_content = Path("frontend/app.js").read_text()

# Filter out the API_URL line from app.js
js_lines = [line for line in js_content.split('\n') if not line.startswith('const API_URL')]
js_filtered = '\n'.join(js_lines)

# Inline everything for HF Spaces
full_html = f"""
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>whoami.dataviz</title>
    <style>{css_content}</style>
    <script src="https://unpkg.com/deck.gl@8.9.35/dist.min.js"></script>
</head>
<body>
    <div id="container">
        <div id="controls">
            <h1>whoami.dataviz</h1>
            <p class="subtitle">"Give me a person's browsing history and I'll tell you who they are."</p>
            <div class="control-group">
                <label>Projection:</label>
                <button id="btn-umap" class="active">UMAP</button>
                <button id="btn-tsne">t-SNE</button>
            </div>
            <div class="control-group">
                <label>Search:</label>
                <input type="text" id="search" placeholder="repo name...">
            </div>
            <div id="stats"></div>
            <div id="legend"></div>
        </div>
        <div id="map"></div>
        <div id="tooltip"></div>
    </div>
    <script>
        const API_URL = 'coords.json';
        {js_filtered}
    </script>
</body>
</html>
"""

with gr.Blocks() as demo:
    gr.HTML(full_html)

demo.launch()
