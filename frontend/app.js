// whoami.dataviz - Deck.gl visualization

const API_URL = '/coords';

const COLORS = {
    clusters: [
        [59, 130, 246],   // blue
        [16, 185, 129],   // green
        [245, 158, 11],   // amber
        [239, 68, 68],    // red
        [139, 92, 246],   // purple
        [236, 72, 153],   // pink
        [20, 184, 166],   // teal
        [249, 115, 22],   // orange
        [132, 204, 22],   // lime
        [6, 182, 212],    // cyan
    ],
    noise: [128, 128, 128],
};

let data = null;
let currentProjection = 'umap';
let searchTerm = '';
let deckgl = null;

async function loadData() {
    try {
        const response = await fetch(API_URL);
        if (!response.ok) {
            throw new Error(`HTTP ${response.status}`);
        }
        data = await response.json();
        updateStats();
        updateLegend();
        render();
    } catch (error) {
        console.error('Failed to load data:', error);
        document.getElementById('stats').innerHTML = `
            <span style="color: #ef4444;">Failed to load data.</span><br>
            Run the pipeline first:<br>
            <code style="color: #888;">python -m pipeline.main</code>
        `;
    }
}

function getColor(repo) {
    if (searchTerm && !repo.name.toLowerCase().includes(searchTerm.toLowerCase())) {
        return [50, 50, 50, 100];
    }

    if (repo.cluster_id === -1) {
        return [...COLORS.noise, 102];  // 40% opacity
    }

    const colorIndex = repo.cluster_id % COLORS.clusters.length;
    return [...COLORS.clusters[colorIndex], 180];  // 70% opacity
}

function getPosition(repo) {
    return currentProjection === 'umap' ? repo.umap : repo.tsne;
}

function getRadius(repo) {
    if (searchTerm && repo.name.toLowerCase().includes(searchTerm.toLowerCase())) {
        return 0.15;
    }
    return 0.08;
}

function render() {
    if (!data) return;

    const layer = new deck.PointCloudLayer({
        id: 'repos',
        data: data.repos,
        getPosition,
        getColor,
        pointSize: 4,
        sizeUnits: 'pixels',
        pickable: true,
        onHover: handleHover,
        updateTriggers: {
            getColor: [searchTerm],
            getPosition: [currentProjection],
        },
        transitions: {
            getPosition: 500,
            getColor: 200,
        },
    });

    if (!deckgl) {
        deckgl = new deck.DeckGL({
            container: 'map',
            views: new deck.OrbitView({
                orbitAxis: 'Y',
            }),
            initialViewState: {
                target: [0, 0, 0],
                zoom: 4,
                rotationX: 45,
                rotationOrbit: 30,
            },
            controller: {
                scrollZoom: { speed: 2.5, smooth: true },
                dragRotate: true,
                dragPan: true,
            },
            layers: [layer],
        });
    } else {
        deckgl.setProps({ layers: [layer] });
    }
}

function handleHover({ object, x, y }) {
    const tooltip = document.getElementById('tooltip');

    if (object) {
        tooltip.style.display = 'block';
        tooltip.style.left = (x + 15) + 'px';
        tooltip.style.top = (y + 15) + 'px';

        const confidence = object.confidence ? `${Math.round(object.confidence * 100)}% confidence` : '';

        tooltip.innerHTML = `
            <div class="name">${object.name}</div>
            <div class="cluster">${object.cluster_name}</div>
            ${confidence ? `<div class="confidence">${confidence}</div>` : ''}
        `;
    } else {
        tooltip.style.display = 'none';
    }
}

function updateStats() {
    const stats = document.getElementById('stats');
    const clusters = data.clusters.length;
    const repos = data.repos.length;
    const noise = data.repos.filter(r => r.cluster_id === -1).length;

    stats.innerHTML = `
        <strong>${repos}</strong> repos<br>
        <strong>${clusters}</strong> clusters<br>
        <strong>${noise}</strong> unclustered
    `;
}

function updateLegend() {
    const legend = document.getElementById('legend');

    const items = data.clusters
        .sort((a, b) => b.size - a.size)
        .slice(0, 8)
        .map(cluster => {
            const colorIndex = cluster.id % COLORS.clusters.length;
            const color = COLORS.clusters[colorIndex];
            return `
                <div class="legend-item">
                    <div class="legend-color" style="background: rgb(${color.join(',')})"></div>
                    <span class="legend-label">${cluster.name} (${cluster.size})</span>
                </div>
            `;
        })
        .join('');

    const noiseCount = data.repos.filter(r => r.cluster_id === -1).length;
    const noiseItem = noiseCount > 0 ? `
        <div class="legend-item">
            <div class="legend-color" style="background: rgb(${COLORS.noise.join(',')}); opacity: 0.4"></div>
            <span class="legend-label">unclustered (${noiseCount})</span>
        </div>
    ` : '';

    legend.innerHTML = items + noiseItem;
}

function setProjection(proj) {
    currentProjection = proj;
    document.getElementById('btn-umap').classList.toggle('active', proj === 'umap');
    document.getElementById('btn-tsne').classList.toggle('active', proj === 'tsne');
    render();
}

// Event listeners
document.getElementById('btn-umap').addEventListener('click', () => setProjection('umap'));
document.getElementById('btn-tsne').addEventListener('click', () => setProjection('tsne'));
document.getElementById('search').addEventListener('input', (e) => {
    searchTerm = e.target.value;
    render();
});

// Keyboard shortcuts
document.addEventListener('keydown', (e) => {
    if (e.target.tagName === 'INPUT') return;

    if (e.key === 'u' || e.key === 'U') setProjection('umap');
    if (e.key === 't' || e.key === 'T') setProjection('tsne');
    if (e.key === '/' || e.key === 's') {
        e.preventDefault();
        document.getElementById('search').focus();
    }
    if (e.key === 'Escape') {
        document.getElementById('search').value = '';
        searchTerm = '';
        document.getElementById('search').blur();
        render();
    }
});

// Initialize
loadData();
