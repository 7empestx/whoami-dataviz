"""Hugging Face Spaces app - serves the visualization."""
import gradio as gr
import json
import re
from pathlib import Path
from pipeline.synthetic import generate_synthetic_data

# Generate synthetic data
data = generate_synthetic_data(n_repos=250, seed=42)
data_json = json.dumps(data)

# Read the frontend files
css_content = Path("frontend/style.css").read_text()
js_content = Path("frontend/app.js").read_text()

# Remove API_URL line
js_content = re.sub(r"const API_URL = .*;\n", "", js_content)

# Remove the original loadData function (async function loadData() { ... })
js_content = re.sub(
    r"async function loadData\(\) \{[\s\S]*?\n\}\n",
    "",
    js_content
)

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
        console.log('Script starting...');

        // Inline data
        const data = {data_json};
        console.log('Data loaded:', data.repos.length, 'repos');

        // Initialize immediately since data is inline
        function init() {{
            console.log('init() called');
            try {{
                updateStats();
                console.log('updateStats done');
                updateLegend();
                console.log('updateLegend done');
                render();
                console.log('render done');
            }} catch (e) {{
                console.error('Error in init:', e);
            }}
        }}

        {js_content}

        // Replace loadData call with init
        console.log('Calling init...');
        init();
        console.log('Script complete');
    </script>
</body>
</html>
"""

with gr.Blocks() as demo:
    gr.HTML(full_html)

demo.launch()
