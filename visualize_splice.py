import os
import base64
import numpy as np
import pandas as pd
from io import BytesIO
from PIL import Image
import json

from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans
import umap.umap_ as umap

from dash import Dash, dcc, html, Input, Output, State, no_update, clientside_callback
import plotly.graph_objects as go
import plotly.colors

# --- CONFIG ---
INPUT_FILE = 'splice_data_sketchy.npz'
IMAGE_FOLDER = os.path.join('Sketchy', 'images')
NUM_CLUSTERS = 10 

# --- LOAD DATA ---
if not os.path.exists(INPUT_FILE):
    raise FileNotFoundError(f"Could not find {INPUT_FILE}. Please run generate_splice_embeddings.py first.")

print(f"Loading SpLiCE data from {INPUT_FILE}...")
data = np.load(INPUT_FILE)
weights = data['weights']   # Shape (N, 10000)
vocab = data['vocab']       # Shape (10000,)

# Decode strings
filenames = [f.decode('utf-8') if isinstance(f, bytes) else f for f in data['filenames']]
captions = [c.decode('utf-8') if isinstance(c, bytes) else c for c in data['captions']]

N = len(filenames)
print(f"Loaded {N} samples.")

# --- CLUSTERING & REDUCTION ---
# SpLiCE weights are our "embeddings". We use them for clustering/viz.
print(f"Running K-Means Clustering on SpLiCE weights (k={NUM_CLUSTERS})...")
kmeans = KMeans(n_clusters=NUM_CLUSTERS, random_state=42, n_init=10)
cluster_labels = kmeans.fit_predict(weights)

print("Running PCA...")
pca_2d = PCA(n_components=2).fit_transform(weights)

print("Running t-SNE...")
# Metric='cosine' is often better for sparse data, but euclidean is standard for t-SNE
tsne_2d = TSNE(n_components=2, perplexity=min(30, N - 1), metric='cosine', random_state=42).fit_transform(weights)

print("Running UMAP...")
umap_2d = umap.UMAP(n_components=2, metric='cosine', random_state=42).fit_transform(weights)

reductions = {
    "PCA": pca_2d,
    "t-SNE": tsne_2d,
    "UMAP": umap_2d
}

colormap = plotly.colors.qualitative.Bold
point_colors = [colormap[i % len(colormap)] for i in cluster_labels]

# --- DASH APP ---
app = Dash(__name__)

TOOLTIP_STYLE = {
    "padding": "12px",
    "backgroundColor": "white",
    "color": "#333",
    "borderRadius": "8px",
    "width": "260px",
    "whiteSpace": "normal",
    "wordWrap": "break-word",
    "fontFamily": "Arial, sans-serif",
    "boxShadow": "0px 4px 15px rgba(0,0,0,0.2)",
    "border": "1px solid #eee",
    "zIndex": 1000
}

app.layout = html.Div([
    html.H3("SpLiCE Embedding", style={"textAlign": "center", "fontFamily": "Arial, sans-serif", "color": "#333"}),

    html.Div([
        dcc.Dropdown(
            id="algo",
            options=[{"label": k, "value": k} for k in reductions],
            value="UMAP",
            clearable=False,
            style={"width": "200px", "margin": "auto"}
        )
    ], style={"textAlign": "center", "marginBottom": "20px"}),

    dcc.Graph(id="plot", style={"height": "75vh"}, clear_on_unhover=True),
    dcc.Tooltip(id="tooltip", style=TOOLTIP_STYLE),
    
    # Stores for JS callback
    dcc.Store(id='cluster-data-store', data=cluster_labels.tolist()),
    dcc.Store(id='N-store', data=N)

], style={"backgroundColor": "#fafafa", "padding": "20px", "fontFamily": "Arial, sans-serif"})

def encode_image(rel_path):
    path = os.path.join(IMAGE_FOLDER, rel_path)
    try:
        img = Image.open(path).convert("RGB")
        img.thumbnail((200, 200))
        buf = BytesIO()
        img.save(buf, format="JPEG", quality=70)
        return "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode()
    except Exception as e:
        return None

def get_top_concepts(idx, top_k=5):
    """Helper to extract top SpLiCE concepts for tooltip"""
    w = weights[idx]
    top_indices = np.argsort(w)[-top_k:][::-1]
    concepts = []
    for i in top_indices:
        val = w[i]
        if val > 0.001: 
            concepts.append(f"{vocab[i]} ({val:.2f})")
    if not concepts: return ["No strong concepts found"]
    return concepts

# --- CALLBACK 1: UPDATE GRAPH ---
@app.callback(
    Output("plot", "figure"),
    Input("algo", "value")
)
def update_figure(algo):
    coords = reductions[algo]
    
    # Customdata: [Index, ClusterID]
    customdata = np.stack((np.arange(N), cluster_labels), axis=-1).tolist()

    fig = go.Figure()

    # Trace 0: Images (SpLiCE Embeddings)
    fig.add_trace(go.Scatter(
        x=coords[:, 0],
        y=coords[:, 1],
        mode='markers',
        name='Images',
        marker=dict(
            symbol='circle',
            size=9,
            color=point_colors,
            opacity=0.7,
            line=dict(width=1, color='white')
        ),
        customdata=customdata,
        hoverinfo='none'
    ))

    fig.update_layout(
        template="plotly_white",
        hovermode="closest",
        xaxis=dict(showgrid=True, gridcolor='#f0f0f0', showticklabels=True, zeroline=True),
        yaxis=dict(showgrid=True, gridcolor='#f0f0f0', showticklabels=True, zeroline=True),
        margin=dict(l=20, r=20, t=20, b=20),
        legend=dict(title="Clusters (SpLiCE Concepts)")
    )
    return fig

# --- CALLBACK 2: SHOW TOOLTIP ---
@app.callback(
    Output("tooltip", "show"),
    Output("tooltip", "bbox"),
    Output("tooltip", "children"),
    Input("plot", "hoverData")
)
def update_tooltip(hoverData):
    if hoverData is None:
        return False, no_update, no_update

    try:
        pt = hoverData["points"][0]
        cdata = pt.get("customdata", [])
        
        if not cdata or len(cdata) < 2:
            return False, no_update, no_update
            
        idx = cdata[0]
        cluster_id = cdata[1]
        
        real_caption = captions[idx]
        top_concepts = get_top_concepts(idx)
        img_src = encode_image(filenames[idx])

        # Construct Tooltip
        children = [
            html.Div(f"Cluster Group {cluster_id}", style={"fontSize": "10px", "color": "#999", "textAlign": "right", "marginBottom": "4px"}),
            html.Img(src=img_src, style={"width": "100%", "borderRadius": "4px", "marginBottom": "8px"}),
            
            html.Div([
                html.Span("Original Caption: ", style={"fontWeight": "bold", "color": "#555", "fontSize": "11px"}),
                html.Span(real_caption, style={"fontSize": "12px", "lineHeight": "1.3"})
            ]),

            html.Hr(style={"margin": "8px 0", "borderTop": "1px solid #eee"}),

            html.Div([
                html.Span("SpLiCE Concepts:", style={"fontWeight": "bold", "color": "#d62728", "fontSize": "11px"}),
                html.Ul([html.Li(c) for c in top_concepts], style={"paddingLeft": "15px", "margin": "4px 0", "fontSize": "11px"})
            ])
        ]

        return True, pt["bbox"], children
    except Exception:
        return False, no_update, no_update

# --- CALLBACK 3: INSTANT HIGHLIGHTING (JS) ---
# Modified to work with single trace (Images) instead of 2 traces
clientside_callback(
    """
    function(hoverData, figure, clusterLabels, N) {
        // 1. Un-hover
        if (!hoverData || !figure) {
            const newOpacity = Array(N).fill(0.7);
            
            const newFigure = JSON.parse(JSON.stringify(figure));
            newFigure.data[0].marker.opacity = newOpacity;
            return newFigure;
        }

        const pt = hoverData.points[0];
        
        // Safety Check
        if (!pt.customdata || pt.customdata.length < 2) {
            return window.dash_clientside.no_update;
        }
        
        const targetCluster = pt.customdata[1];
        const newOpacity = [];

        for (let i = 0; i < N; i++) {
            newOpacity.push(clusterLabels[i] === targetCluster ? 1.0 : 0.1);
        }

        const newFigure = JSON.parse(JSON.stringify(figure));
        newFigure.data[0].marker.opacity = newOpacity;

        return newFigure;
    }
    """,
    Output('plot', 'figure', allow_duplicate=True),
    Input('plot', 'hoverData'),
    State('plot', 'figure'),
    State('cluster-data-store', 'data'),
    State('N-store', 'data'),
    prevent_initial_call=True
)

if __name__ == "__main__":
    print("Dash App is running... Open http://127.0.0.1:8050/")
    app.run(debug=True)