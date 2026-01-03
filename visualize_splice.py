import os
import base64
import numpy as np
import pandas as pd
from io import BytesIO
from PIL import Image
import json

# pip install scikit-learn umap-learn dash pandas plotly
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans
import umap.umap_ as umap

from dash import Dash, dcc, html, Input, Output, State, no_update, clientside_callback
import plotly.graph_objects as go
import plotly.colors

# --------------------------------------------------
# CONFIG
# --------------------------------------------------
INPUT_FILE = "splice_embeddings_100.npz"
VOCAB_PATH = os.path.join("vocab", "laion.txt")
NUM_CLUSTERS = 10

# --------------------------------------------------
# LOAD DATA
# --------------------------------------------------
if not os.path.exists(INPUT_FILE):
    raise FileNotFoundError(f"Could not find {INPUT_FILE}")

print(f"Loading SpLiCE data from {INPUT_FILE}...")
data = np.load(INPUT_FILE, allow_pickle=True)

# SpLiCE sparse embeddings
weights = data["sparse"]              # (N, num_concepts)

# Image paths (absolute or relative)
filenames = data["image_paths"]

# Load vocabulary
with open(VOCAB_PATH, "r", encoding="utf-8") as f:
    vocab = [l.strip() for l in f.readlines()]

# Keep only vocab size actually used
vocab = vocab[-weights.shape[1]:]

# Use folder name as class / caption (Sketchy-style)
captions = [os.path.basename(os.path.dirname(p)) for p in filenames]

N = len(filenames)
print(f"Loaded {N} samples.")
print("weights shape:", weights.shape)

# --- CLUSTERING & REDUCTION (OPTIMIZED) ---

# 1. PCA Reduction (The Speed Fix)
print("Running PCA reduction (10000 -> 50 components)...")
pca_50 = PCA(n_components=50, random_state=42).fit_transform(weights)

# 2. Clustering (on 50D data)
print(f"Running K-Means Clustering (k={NUM_CLUSTERS})...")
kmeans = KMeans(n_clusters=NUM_CLUSTERS, random_state=42, n_init=10)
cluster_labels = kmeans.fit_predict(pca_50)

# 3. Dimensionality Reduction (on 50D data)
print("Running 2D PCA...")
pca_2d = PCA(n_components=2).fit_transform(weights)

print("Running t-SNE (on 50D data)...")
# metric='cosine' is good for semantic vectors, but 'euclidean' is faster/standard for t-SNE
tsne_2d = TSNE(n_components=2, perplexity=min(30, N - 1), metric='cosine', random_state=42).fit_transform(pca_50)

print("Running UMAP (on 50D data)...")
umap_2d = umap.UMAP(n_components=2, metric='cosine', random_state=42).fit_transform(pca_50)

reductions = {
    "UMAP": umap_2d,
    "t-SNE": tsne_2d,
    "PCA": pca_2d
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
    html.H3("SpLiCE Embedding Explorer", style={"textAlign": "center", "fontFamily": "Arial, sans-serif", "color": "#333"}),

    html.Div([
        html.Label("Projection Algorithm: ", style={"fontWeight": "bold", "marginRight": "10px"}),
        dcc.Dropdown(
            id="algo",
            options=[{"label": k, "value": k} for k in reductions],
            value="UMAP",
            clearable=False,
            style={"width": "200px", "display": "inline-block", "verticalAlign": "middle"}
        )
    ], style={"textAlign": "center", "marginBottom": "20px"}),

    dcc.Graph(id="plot", style={"height": "75vh"}, clear_on_unhover=True),
    dcc.Tooltip(id="tooltip", style=TOOLTIP_STYLE),
    
    # Stores for JS callback
    dcc.Store(id='cluster-data-store', data=cluster_labels.tolist()),
    dcc.Store(id='N-store', data=N)

], style={"backgroundColor": "#fafafa", "padding": "20px", "fontFamily": "Arial, sans-serif"})

def encode_image(path):
    try:
        if not os.path.exists(path):
            return None
        img = Image.open(path).convert("RGB")
        img.thumbnail((200, 200))
        buf = BytesIO()
        img.save(buf, format="JPEG", quality=70)
        return "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode()
    except Exception as e:
        print("Image load error:", e)
        return None


def get_top_concepts(idx, top_k=5):
    """Helper to extract top SpLiCE concepts for tooltip"""
    w = weights[idx]
    # Get indices of top k weights
    top_indices = np.argsort(w)[-top_k:][::-1]
    concepts = []
    for i in top_indices:
        val = w[i]
        # Only show concepts with meaningful weight
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
    fig.add_trace(go.Scattergl(  # Scattergl is faster for many points
        x=coords[:, 0],
        y=coords[:, 1],
        mode='markers',
        name='Images',
        marker=dict(
            symbol='circle',
            size=8,
            color=point_colors,
            opacity=0.7,
            line=dict(width=0.5, color='white')
        ),
        customdata=customdata,
        hoverinfo='none'
    ))

    fig.update_layout(
        template="plotly_white",
        hovermode="closest",
        xaxis=dict(showgrid=True, gridcolor='#f0f0f0', showticklabels=False, zeroline=False),
        yaxis=dict(showgrid=True, gridcolor='#f0f0f0', showticklabels=False, zeroline=False),
        margin=dict(l=20, r=20, t=20, b=20),
        legend=dict(title="Clusters")
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
                html.Span("Class: ", style={"fontWeight": "bold", "color": "#555", "fontSize": "11px"}),
                html.Span(real_caption, style={"fontSize": "12px", "lineHeight": "1.3"})
            ]),

            html.Hr(style={"margin": "8px 0", "borderTop": "1px solid #eee"}),

            html.Div([
                html.Span("Top Concepts:", style={"fontWeight": "bold", "color": "#d62728", "fontSize": "11px"}),
                html.Ul([html.Li(c) for c in top_concepts], style={"paddingLeft": "15px", "margin": "4px 0", "fontSize": "11px"})
            ])
        ]

        return True, pt["bbox"], children
    except Exception:
        return False, no_update, no_update

# --- CALLBACK 3: INSTANT HIGHLIGHTING (JS) ---
clientside_callback(
    """
    function(hoverData, figure, clusterLabels, N) {
        // 1. Un-hover
        if (!hoverData || !figure) {
            // Restore default opacity if no hover
            // We need to return the figure with restored opacity
            // Check if we need to clone.
            // Actually, returning no_update is better if we just want to reset,
            // but here we want to reset visual state.
            
            const newFigure = JSON.parse(JSON.stringify(figure));
            if (newFigure.data[0].marker.opacity && Array.isArray(newFigure.data[0].marker.opacity)) {
                 // Reset to single value or full array of 0.7
                 newFigure.data[0].marker.opacity = 0.7;
                 return newFigure;
            }
            return window.dash_clientside.no_update;
        }

        const pt = hoverData.points[0];
        
        if (!pt.customdata || pt.customdata.length < 2) {
            return window.dash_clientside.no_update;
        }
        
        const targetCluster = pt.customdata[1];
        const newOpacity = [];

        // Simple loop to dim non-cluster points
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